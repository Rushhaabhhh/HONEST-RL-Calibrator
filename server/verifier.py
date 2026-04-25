"""Answer verification with domain-specific routing."""

import math
from typing import Any, Dict, Optional

from data.verifiers.code_verifier import verify_code_answer
from data.verifiers.logic_verifier import verify_logic_answer
from data.verifiers.math_verifier import verify_math_answer


def verify_answer(
    agent_answer: str,
    ground_truth: str,
    domain: Optional[str] = None,
    verification_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if agent_answer matches ground_truth, using domain-specific rules."""
    if not agent_answer or not agent_answer.strip():
        return False
        
    verification_metadata = verification_metadata or {}

    try:
        if domain == "math":
            return verify_math_answer(agent_answer, ground_truth)
        elif domain == "code":
            return verify_code_answer(agent_answer, verification_metadata)
        elif domain == "logic":
            passed, _acc = verify_logic_answer(agent_answer, ground_truth, verification_metadata)
            return passed
        else:
            # Fallback to simple normalization if domain is unknown or None
            return _normalize(agent_answer) == _normalize(ground_truth)
    except Exception:
        # Defensive fallback
        try:
            return _normalize(agent_answer) == _normalize(ground_truth)
        except Exception:
            return False


def _normalize(s: str) -> str:
    """Normalize a string for fallback comparison."""
    s = s.strip().lower().replace(",", "")
    try:
        f = float(s)
        if not math.isfinite(f):
            return s
        if abs(f) > 1e15:
            return f"{f:g}"
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        pass
    return s
