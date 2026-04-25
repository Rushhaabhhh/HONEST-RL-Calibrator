"""Singleton accessor and module-level generate() shims for all three domains.

This module provides:
  - ``get_sampler()``  — returns the process-wide singleton UnifiedSampler
  - ``math_generate(difficulty, seed)``   → (str, str, str)
  - ``code_generate(difficulty, seed)``   → (str, str, str)
  - ``logic_generate(difficulty, seed)``  → (str, str, str)

Each generate function returns ``(question, canonical_answer, problem_id)``.
``problem_id`` is the stable ID stored on each ``UnifiedProblem``; for
procedural logic problems (difficulties 1-2) it is a synthetic ID prefixed
with ``procedural_logic_`` so the reward layer can route to the right
verifier.

They are re-exported from the three thin shim modules:
  data/sampler/math_gen_adapter.py
  data/sampler/code_gen_adapter.py
  data/sampler/logic_gen_adapter.py
so the import line in ``server/environment.py`` only needs to change once
per domain.
"""

from __future__ import annotations

from typing import Optional, Tuple

from data.sampler.unified_sampler import (
    generate_code,
    generate_logic,
    generate_math,
    get_sampler,
)

__all__ = [
    "get_sampler",
    "math_generate",
    "code_generate",
    "logic_generate",
]


def math_generate(
    difficulty: int,
    seed: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Drop-in replacement for ``server.generators.math_gen.generate``."""
    return generate_math(difficulty, seed)


def code_generate(
    difficulty: int,
    seed: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Drop-in replacement for ``server.generators.code_gen.generate``."""
    return generate_code(difficulty, seed)


def logic_generate(
    difficulty: int,
    seed: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Drop-in replacement for ``server.generators.logic_gen.generate``.

    Routes to the procedural generator for difficulties 1-2 and the
    curated ZebraLogic dataset for difficulties 3-5.
    """
    return generate_logic(difficulty, seed)
