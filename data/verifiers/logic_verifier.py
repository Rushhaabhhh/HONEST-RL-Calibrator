"""Verifier for logic (ZebraLogic-style) puzzle answers.

Exposes :func:`verify_logic_answer`, which parses a model's JSON answer,
computes per-cell accuracy against the canonical solution, and returns a
*(passes_threshold, cell_accuracy)* pair.

Scoring
-------
* Parse the model output as JSON.  On failure → ``(False, 0.0)``.
* For each cell ``(house, feature)`` in the canonical answer, check whether
  the model's value matches (case-insensitive, whitespace-stripped).
* ``cell_accuracy = correct_cells / total_cells``
* Return ``(cell_accuracy >= 0.9, cell_accuracy)``

House-key normalisation
-----------------------
The verifier maps all of the following forms to the same integer index:
``"House 1"``, ``"house 1"``, ``"house_1"``, ``"1"``, ``1``
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# House key normalisation
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d+")


def _house_index(key: Union[str, int]) -> Optional[int]:
    """Extract the house number from a key in any expected format.

    Returns ``None`` if no integer can be found.
    """
    if isinstance(key, int):
        return key
    s = str(key)
    m = _DIGIT_RE.search(s)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------


def _norm(v: Any) -> str:
    """Normalise a value for comparison: lower-case, strip whitespace."""
    return str(v).strip().lower()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort: find and parse the first ``{...}`` block in *text*."""
    text = text.strip()
    # Try whole string first (common case when model output is clean JSON)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to searching for a JSON object
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_logic_answer(
    model_answer: str,
    canonical_answer: Union[Dict[str, Any], str],
    verification_metadata: Dict[str, Any],
) -> Tuple[bool, float]:
    """Check a model's answer against the canonical solution.

    Parameters
    ----------
    model_answer:
        Raw string output from the model.  Expected to contain a JSON object.
    canonical_answer:
        The ground-truth assignment.  May be a dict (preferred) or a JSON
        string produced by ``UnifiedProblem.to_jsonl`` round-tripping.
    verification_metadata:
        Puzzle metadata (used for ``features`` list if needed).

    Returns
    -------
    (passes_threshold, cell_accuracy) where passes_threshold is
    ``cell_accuracy >= 0.9``.
    """
    # --- Parse canonical answer ---
    if isinstance(canonical_answer, str):
        try:
            canon = json.loads(canonical_answer)
        except (json.JSONDecodeError, ValueError):
            return False, 0.0
    elif isinstance(canonical_answer, dict):
        canon = canonical_answer
    else:
        return False, 0.0

    if not canon:
        return False, 0.0

    # --- Parse model answer ---
    if not isinstance(model_answer, str):
        return False, 0.0

    parsed = _extract_json(model_answer)
    if parsed is None or not isinstance(parsed, dict):
        return False, 0.0

    # --- Build normalised lookup: house_index → {feature_lower: value_lower}
    model_map: Dict[int, Dict[str, str]] = {}
    for key, attrs in parsed.items():
        idx = _house_index(key)
        if idx is None or not isinstance(attrs, dict):
            continue
        model_map[idx] = {k.strip().lower(): _norm(v) for k, v in attrs.items()}

    # --- Score ---
    correct = 0
    total = 0
    for house_key, attrs in canon.items():
        canon_idx = _house_index(house_key)
        if canon_idx is None or not isinstance(attrs, dict):
            continue
        model_attrs = model_map.get(canon_idx, {})
        for feat, canon_val in attrs.items():
            total += 1
            model_val = model_attrs.get(feat.strip().lower())
            if model_val is not None and model_val == _norm(canon_val):
                correct += 1

    if total == 0:
        return False, 0.0

    accuracy = correct / total
    return accuracy >= 0.9, round(accuracy, 6)
