#!/usr/bin/env python3
"""
Unified KD client with backend selection, caching, and async prefetch.
"""
import os
import hashlib
import json
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from kd.kd_backend import TeacherBackend, TeacherOutput
from kd.venice_backend import VeniceBackend
from kd.vllm_backend import VLLMBackend


class KDClient:
    """
    Unified knowledge distillation client.

    Selects backend based on TEACHER_BACKEND environment variable:
    - venice: Venice API (remote)
    - vllm-local: Local vLLM in-process
    - vllm-remote: Remote vLLM API

    Includes on-disk caching for KD distributions.
    """

    def __init__(
        self,
        backend_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl_days: int = 30,  # Cache expiry in days
        **backend_kwargs,
    ):
        """
        Initialize KD client.

        Args:
            backend_type: Backend type (venice, vllm-local, vllm-remote)
                         Defaults to TEACHER_BACKEND env var
            cache_dir: Directory for KD cache (default: ./cache)
            enable_cache: Enable KD caching
            cache_ttl_days: Cache TTL in days (default: 30)
            **backend_kwargs: Additional arguments for backend initialization
        """
        load_dotenv()

        # Determine backend type
        self.backend_type = backend_type or os.getenv("TEACHER_BACKEND", "venice")
        self.cache_ttl_days = cache_ttl_days

        # Initialize backend
        print(f"Initializing KD client with backend: {self.backend_type}")

        if self.backend_type == "venice":
            self.backend = VeniceBackend(**backend_kwargs)
        elif self.backend_type == "vllm-local":
            # Local vLLM
            model_path = backend_kwargs.get("model_path") or os.getenv("VLLM_LOCAL_MODEL")
            if not model_path:
                raise ValueError("VLLM_LOCAL_MODEL not set for vllm-local backend")
            self.backend = VLLMBackend(model_path=model_path, **backend_kwargs)
        elif self.backend_type == "vllm-remote":
            # Remote vLLM
            base_url = backend_kwargs.get("base_url") or os.getenv("VLLM_REMOTE_URL")
            if not base_url:
                raise ValueError("VLLM_REMOTE_URL not set for vllm-remote backend")
            self.backend = VLLMBackend(base_url=base_url, **backend_kwargs)
        else:
            raise ValueError(
                f"Unknown backend type: {self.backend_type}. "
                f"Valid options: venice, vllm-local, vllm-remote"
            )

        # Initialize cache
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir or "./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_db_path = self.cache_dir / "kd_cache.sqlite"
        self.cache_hits = 0
        self.cache_misses = 0
        self.trained_from_cache = 0  # Counter for steps trained from cache (e.g., when rate limited)

        # Async prefetch support
        self.prefetch_queue = {}  # Cache key -> Future[TeacherOutput]
        self.executor = ThreadPoolExecutor(max_workers=4)  # For async I/O
        self.prefetch_enabled = True

        # Adaptive load handling
        self.max_concurrent_requests = 8  # Semaphore limit
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.adaptive_top_logprobs = 20  # Start with 20
        self.min_top_logprobs = 10  # Minimum when rate limited
        self.max_top_logprobs = 20  # Maximum
        self.rate_limit_count = 0
        self.successful_window_count = 0
        self.windows_before_restore = 10  # Restore to 20 after N successful windows

        if self.enable_cache:
            self._init_cache_db()

        print(f"✅ KD client initialized with async prefetch and adaptive load handling")

    def _init_cache_db(self):
        """Initialize SQLite cache database with TTL support."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # Create cache table with additional metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kd_cache (
                cache_key TEXT PRIMARY KEY,
                teacher_output TEXT NOT NULL,
                backend_type TEXT NOT NULL,
                model_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for efficient TTL cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON kd_cache(timestamp)
        """)

        conn.commit()
        conn.close()

        # Clean expired entries on init
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self):
        """Remove cache entries older than TTL."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM kd_cache
                WHERE julianday('now') - julianday(timestamp) > ?
            """, (self.cache_ttl_days,))

            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"🧹 Cleaned {deleted_count} expired cache entries (>{self.cache_ttl_days} days old)")

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Cache cleanup error: {e}")

    def _get_cache_key(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Generate cache key for request.

        Cache key includes:
        - Backend type (venice, vllm-local, vllm-remote)
        - Model ID (for version tracking)
        - Quantization type (affects output distributions)
        - All request parameters

        This ensures cache invalidation when model or backend changes.

        Args:
            messages: Message list
            max_tokens: Max tokens
            top_logprobs: Top logprobs count
            temperature: Temperature
            top_p: Top-p value

        Returns:
            Cache key (SHA256 hash)
        """
        # Get model ID based on backend
        model_id = None
        if self.backend_type == "venice":
            model_id = os.getenv("VENICE_MODEL", "qwen3-next-80b")
        elif self.backend_type in ["vllm-local", "vllm-remote"]:
            model_id = os.getenv("VLLM_LOCAL_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
            # Include quantization in key for vLLM
            quantization = os.getenv("VLLM_QUANTIZATION", "gptq")
            model_id = f"{model_id}:{quantization}"

        # Include backend type and model in key
        key_data = {
            "backend": self.backend_type,
            "model_id": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
            "top_p": top_p,
            "cache_version": "v2",  # Increment when changing cache format
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[TeacherOutput]:
        """Get cached output and update access stats."""
        if not self.enable_cache:
            return None

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT teacher_output FROM kd_cache WHERE cache_key = ?",
                (cache_key,)
            )

            row = cursor.fetchone()

            if row:
                self.cache_hits += 1

                # Update access stats
                cursor.execute("""
                    UPDATE kd_cache
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()

                # Deserialize
                data = json.loads(row[0])
                conn.close()
                return TeacherOutput(
                    tokens=data["tokens"],
                    logprobs=data["logprobs"],
                    normalized_probs=data["normalized_probs"],
                )
            else:
                self.cache_misses += 1
                conn.close()
                return None

        except Exception as e:
            print(f"Cache read error: {e}")
            self.cache_misses += 1
            return None

    def _save_to_cache(self, cache_key: str, output: TeacherOutput):
        """Save output to cache with metadata."""
        if not self.enable_cache:
            return

        try:
            # Serialize
            data = {
                "tokens": output.tokens,
                "logprobs": output.logprobs,
                "normalized_probs": output.normalized_probs,
            }

            # Get model ID for metadata
            model_id = None
            if self.backend_type == "venice":
                model_id = os.getenv("VENICE_MODEL", "qwen3-next-80b")
            elif self.backend_type in ["vllm-local", "vllm-remote"]:
                model_id = os.getenv("VLLM_LOCAL_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO kd_cache
                (cache_key, teacher_output, backend_type, model_id)
                VALUES (?, ?, ?, ?)
            """, (cache_key, json.dumps(data), self.backend_type, model_id))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Cache write error: {e}")

    def _handle_rate_limit(self):
        """Handle rate limiting by reducing top_logprobs."""
        self.rate_limit_count += 1
        if self.adaptive_top_logprobs > self.min_top_logprobs:
            self.adaptive_top_logprobs = self.min_top_logprobs
            print(f"⚠️  Rate limited! Reducing top_logprobs to {self.adaptive_top_logprobs}")
        self.successful_window_count = 0  # Reset success counter

    def _handle_success(self):
        """Handle successful request, possibly restoring top_logprobs."""
        self.successful_window_count += 1
        if self.successful_window_count >= self.windows_before_restore:
            if self.adaptive_top_logprobs < self.max_top_logprobs:
                self.adaptive_top_logprobs = self.max_top_logprobs
                print(f"✅ Restored top_logprobs to {self.adaptive_top_logprobs} after {self.windows_before_restore} successful requests")
            self.successful_window_count = 0

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error (429)."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "too many requests" in error_str

    def get_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        top_logprobs: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Get logprobs from teacher (with caching and adaptive load handling).

        Args:
            messages: Message list
            max_tokens: Max tokens to generate
            top_logprobs: Top logprobs per position (None = use adaptive)
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            TeacherOutput with sparse distributions
        """
        # Use adaptive top_logprobs if not specified
        if top_logprobs is None:
            top_logprobs = self.adaptive_top_logprobs

        # Check cache
        cache_key = self._get_cache_key(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )

        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self._handle_success()  # Successful (from cache)
            return cached

        # Call backend
        try:
            output = self.backend.get_logprobs(
                messages=messages,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                temperature=temperature,
                top_p=top_p,
            )

            # Save to cache
            self._save_to_cache(cache_key, output)

            self._handle_success()  # Successful
            return output

        except Exception as e:
            # Check if rate limit error
            if self._is_rate_limit_error(e):
                self._handle_rate_limit()

            # If API fails, try to return cached version if available
            print(f"⚠️  Backend request failed: {e}")
            print(f"   Falling back to cache if available...")

            # Force cache lookup even if we already tried
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                print(f"   ✅ Using cached response")
                self.trained_from_cache += 1
                return cached
            else:
                print(f"   ❌ No cached response available")
                raise

    def get_teacher_distribution(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
    ) -> TeacherOutput:
        """
        Convenience method for single prompt.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens
            temperature: Temperature
            top_p: Top-p
            top_logprobs: Top logprobs count

        Returns:
            TeacherOutput
        """
        messages = [{"role": "user", "content": prompt}]
        return self.get_logprobs(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )

    def prefetch_async(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
    ):
        """
        Prefetch a KD distribution asynchronously.
        Starts fetching in background without blocking.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens
            temperature: Temperature
            top_p: Top-p
            top_logprobs: Top logprobs count
        """
        if not self.prefetch_enabled:
            return

        messages = [{"role": "user", "content": prompt}]

        # Generate cache key
        cache_key = self._get_cache_key(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )

        # Check if already in cache or prefetch queue
        if cache_key in self.prefetch_queue:
            return

        cached = self._get_from_cache(cache_key)
        if cached is not None:
            # Already cached, store directly
            self.prefetch_queue[cache_key] = cached
            return

        # Submit async fetch to executor
        future = self.executor.submit(
            self._fetch_with_cache,
            messages,
            max_tokens,
            top_logprobs,
            temperature,
            top_p,
            cache_key,
        )
        self.prefetch_queue[cache_key] = future

    def _fetch_with_cache(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: float,
        cache_key: str,
    ) -> Optional[TeacherOutput]:
        """Internal method to fetch with caching (for executor)."""
        try:
            output = self.backend.get_logprobs(
                messages=messages,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                temperature=temperature,
                top_p=top_p,
            )
            self._save_to_cache(cache_key, output)
            return output
        except Exception as e:
            # If fetch fails, try cache
            print(f"Prefetch failed: {e}, trying cache...")
            cached = self._get_from_cache(cache_key)
            if cached:
                self.trained_from_cache += 1
            return cached

    def get_prefetched(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
        timeout: float = 10.0,
    ) -> Optional[TeacherOutput]:
        """
        Get a prefetched KD distribution.
        If not yet ready, waits up to timeout seconds.
        If rate limited or failed, returns cached version if available.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens
            temperature: Temperature
            top_p: Top-p
            top_logprobs: Top logprobs count
            timeout: Max wait time in seconds

        Returns:
            TeacherOutput or None if not available
        """
        messages = [{"role": "user", "content": prompt}]

        cache_key = self._get_cache_key(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )

        # Check prefetch queue
        if cache_key in self.prefetch_queue:
            result = self.prefetch_queue[cache_key]

            # If it's already a TeacherOutput (cached), return directly
            if isinstance(result, TeacherOutput):
                del self.prefetch_queue[cache_key]
                return result

            # Otherwise it's a Future, wait for it
            try:
                output = result.result(timeout=timeout)
                del self.prefetch_queue[cache_key]
                return output
            except Exception as e:
                print(f"Failed to get prefetched result: {e}")
                # Fall through to direct fetch

        # Not prefetched, fetch directly
        return self.get_teacher_distribution(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
        )

    def clear_prefetch_queue(self):
        """Clear the prefetch queue."""
        self.prefetch_queue.clear()

    def get_cache_stats(self) -> Dict:
        """
        Get cache and load statistics.

        Returns:
            Dict with hits, misses, hit rate, and adaptive load metrics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "trained_from_cache": self.trained_from_cache,
            "rate_limit_count": self.rate_limit_count,
            "current_top_logprobs": self.adaptive_top_logprobs,
            "successful_window_count": self.successful_window_count,
        }

    def print_cache_stats(self):
        """Print cache and load statistics."""
        stats = self.get_cache_stats()
        print(f"\nKD Cache & Load Statistics:")
        print(f"  Cache hits: {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        print(f"  Trained from cache: {stats['trained_from_cache']}")
        print(f"  Rate limit events: {stats['rate_limit_count']}")
        print(f"  Current top_logprobs: {stats['current_top_logprobs']}")
        print(f"  Successful window: {stats['successful_window_count']}/{self.windows_before_restore}")


def test_client():
    """Test KD client with current backend."""
    print("Testing KD Client...")
    print(f"Backend: {os.getenv('TEACHER_BACKEND', 'venice')}")

    client = KDClient()

    output = client.get_teacher_distribution(
        prompt="Say hello in one word",
        max_tokens=5,
        temperature=1.0,
        top_logprobs=5,
    )

    print(f"\nGenerated tokens: {output.tokens}")
    print("\nTop probabilities per position:")
    for i, (token, probs) in enumerate(zip(output.tokens, output.normalized_probs)):
        print(f"\nPosition {i} (token: '{token}'):")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for tok, prob in sorted_probs:
            print(f"  '{tok}': {prob:.4f}")

    client.print_cache_stats()


if __name__ == "__main__":
    test_client()
