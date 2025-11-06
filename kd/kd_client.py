#!/usr/bin/env python3
"""
Unified KD client with backend selection and caching.
"""
import os
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
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
        **backend_kwargs,
    ):
        """
        Initialize KD client.

        Args:
            backend_type: Backend type (venice, vllm-local, vllm-remote)
                         Defaults to TEACHER_BACKEND env var
            cache_dir: Directory for KD cache (default: ./cache)
            enable_cache: Enable KD caching
            **backend_kwargs: Additional arguments for backend initialization
        """
        load_dotenv()

        # Determine backend type
        self.backend_type = backend_type or os.getenv("TEACHER_BACKEND", "venice")

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

        if self.enable_cache:
            self._init_cache_db()

        print(f"✅ KD client initialized")

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kd_cache (
                cache_key TEXT PRIMARY KEY,
                teacher_output TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

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

        Args:
            messages: Message list
            max_tokens: Max tokens
            top_logprobs: Top logprobs count
            temperature: Temperature
            top_p: Top-p value

        Returns:
            Cache key (SHA256 hash)
        """
        # Include backend type and all params in key
        key_data = {
            "backend": self.backend_type,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
            "top_p": top_p,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[TeacherOutput]:
        """Get cached output."""
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
            conn.close()

            if row:
                self.cache_hits += 1
                # Deserialize
                data = json.loads(row[0])
                return TeacherOutput(
                    tokens=data["tokens"],
                    logprobs=data["logprobs"],
                    normalized_probs=data["normalized_probs"],
                )
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            print(f"Cache read error: {e}")
            self.cache_misses += 1
            return None

    def _save_to_cache(self, cache_key: str, output: TeacherOutput):
        """Save output to cache."""
        if not self.enable_cache:
            return

        try:
            # Serialize
            data = {
                "tokens": output.tokens,
                "logprobs": output.logprobs,
                "normalized_probs": output.normalized_probs,
            }

            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT OR REPLACE INTO kd_cache (cache_key, teacher_output) VALUES (?, ?)",
                (cache_key, json.dumps(data))
            )

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Cache write error: {e}")

    def get_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Get logprobs from teacher (with caching).

        Args:
            messages: Message list
            max_tokens: Max tokens to generate
            top_logprobs: Top logprobs per position
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            TeacherOutput with sparse distributions
        """
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

            return output

        except Exception as e:
            # If API fails, try to return cached version if available
            print(f"⚠️  Backend request failed: {e}")
            print(f"   Falling back to cache if available...")

            # Force cache lookup even if we already tried
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                print(f"   ✅ Using cached response")
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

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, and hit rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
        }

    def print_cache_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        print(f"\nKD Cache Statistics:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")


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
