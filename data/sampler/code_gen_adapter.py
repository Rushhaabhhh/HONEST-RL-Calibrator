"""Shim exposing ``generate()`` for the code domain.

Drop-in replacement for ``server.generators.code_gen``.

Usage in server/environment.py:
    # BEFORE
    from server.generators import code_gen
    self._generators = {"code": code_gen.generate, ...}

    # AFTER
    from data.sampler.code_gen_adapter import generate as code_generate
    self._generators = {"code": code_generate, ...}
"""

from __future__ import annotations

from typing import Optional, Tuple

from data.sampler.environment_adapter import code_generate


def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Return (question, canonical_answer, problem_id) for a code problem.

    Backed by the UnifiedSampler singleton (lazy-loaded on first call).
    """
    return code_generate(difficulty, seed)
