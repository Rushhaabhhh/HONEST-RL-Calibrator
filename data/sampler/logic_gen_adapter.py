"""Shim exposing ``generate()`` for the logic domain.

Drop-in replacement for ``server.generators.logic_gen``.

The canonical_answer for logic problems is a dict in the JSONL; it is
serialized to a JSON string here so the return type is always (str, str),
matching the original generator interface.  The logic verifier re-parses
the JSON string internally.

Usage in server/environment.py:
    # BEFORE
    from server.generators import logic_gen
    self._generators = {"logic": logic_gen.generate, ...}

    # AFTER
    from data.sampler.logic_gen_adapter import generate as logic_generate
    self._generators = {"logic": logic_generate, ...}
"""

from __future__ import annotations

from typing import Optional, Tuple

from data.sampler.environment_adapter import logic_generate


def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """Return (question, canonical_answer, problem_id) for a logic problem.

    Difficulties 1-2 are routed to the procedural generator in
    ``server.generators.logic_gen`` (``problem_id`` prefixed with
    ``procedural_logic_``).  Difficulties 3-5 are sampled from the
    curated ZebraLogic dataset (JSON-grid answers).
    """
    return logic_generate(difficulty, seed)
