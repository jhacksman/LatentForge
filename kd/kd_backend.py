#!/usr/bin/env python3
"""
Abstract base class for teacher backends.
Supports Venice API, local vLLM, and remote vLLM.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TeacherOutput:
    """
    Output from teacher model.

    Attributes:
        tokens: List of generated token strings
        logprobs: Per-position sparse logprob distributions
                  Each entry is a dict mapping token_str -> logprob
        normalized_probs: Per-position normalized probability distributions
                         Each entry is a dict mapping token_str -> probability
    """
    tokens: List[str]
    logprobs: List[Dict[str, float]]  # Per-position sparse map
    normalized_probs: List[Dict[str, float]]  # Per-position normalized probs


class TeacherBackend(ABC):
    """
    Abstract base class for teacher model backends.

    All backends must implement get_logprobs() to return sparse
    top-k token distributions for knowledge distillation.
    """

    @abstractmethod
    def get_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Generate text and return sparse logprob distributions.

        Args:
            messages: OpenAI-style message list [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top tokens to return per position
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            TeacherOutput with tokens and sparse logprob distributions
        """
        pass

    def get_teacher_distribution(
        self,
        prompt: str,
        max_tokens: int,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Convenience method that wraps prompt as a message.

        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top tokens per position
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            TeacherOutput with sparse distributions
        """
        messages = [{"role": "user", "content": prompt}]
        return self.get_logprobs(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )
