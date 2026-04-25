"""Shim exposing ``generate()`` for the math domain.

Drop-in replacement for ``server.generators.math_gen``.

Usage in server/environment.py:
    # BEFORE
    from server.generators import math_gen
    self._generators = {"math": math_gen.generate, ...}

    # AFTER
    from data.sampler.math_gen_adapter import generate as math_generate
    self._generators = {"math": math_generate, ...}
"""

from __future__ import annotations

from typing import Optional, Tuple

from data.sampler.environment_adapter import math_generate


def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Return (question, canonical_answer, problem_id) for a math problem.

    Backed by the UnifiedSampler singleton (lazy-loaded on first call).
    """
    return math_generate(difficulty, seed)
