"""Adaptive difficulty management for the HONEST environment.

Each episode record stored in ``state.episode_history`` is a plain dict:

    {
        "domain": str,          # which domain the question came from
        "correct": bool | None, # None when answer was abstained/malformed
        "difficulty": int,      # difficulty level at the time of the episode
    }

``update_difficulty`` inspects the last ``WINDOW`` records for the active
domain and adjusts ``state.domain_difficulties[domain]`` according to the
rolling-accuracy thresholds, subject to a hysteresis guard that prevents
more than one change per ``HYSTERESIS_EPISODES`` episodes.
"""

from typing import Optional

from models.models import HonestState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW: int = 20            # rolling accuracy window (episodes per domain)
HIGH_THRESHOLD: float = 0.70  # accuracy above this → increase difficulty
LOW_THRESHOLD: float = 0.30   # accuracy below this → decrease difficulty
MIN_DIFFICULTY: int = 1
MAX_DIFFICULTY: int = 5
HYSTERESIS_EPISODES: int = 10  # min episodes between consecutive changes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _domain_records(state: HonestState, domain: str) -> list[dict]:
    """Return the last WINDOW episode records for *domain* from history."""
    records = [r for r in state.episode_history if r.get("domain") == domain]
    return records[-WINDOW:]


def _last_change_episode(state: HonestState, domain: str) -> int:
    """Return the global episode index of the most recent difficulty change for domain.

    We scan episode_history backwards for a record flagged with
    ``"difficulty_changed": True`` for the given domain.
    Returns 0 if no change has ever occurred (epoch 0 = safe to change).
    """
    for idx in range(len(state.episode_history) - 1, -1, -1):
        r = state.episode_history[idx]
        if r.get("domain") == domain and r.get("difficulty_changed"):
            return idx
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rolling_accuracy(state: HonestState, domain: str) -> float:
    """Return the rolling accuracy (0.0–1.0) for *domain* over the last WINDOW episodes.

    Episodes where ``correct`` is ``None`` (abstain / malformed) are treated
    as incorrect for the purpose of accuracy computation.  Returns 0.5 if
    there are no recorded episodes for the domain yet (neutral starting point).
    """
    records = _domain_records(state, domain)
    if not records:
        return 0.5  # neutral default — no change triggered
    correct_count = sum(1 for r in records if r.get("correct") is True)
    return correct_count / len(records)


def update_difficulty(
    state: HonestState,
    last_correctness: Optional[bool],
    domain: Optional[str] = None,
) -> None:
    """Update ``state.domain_difficulties[domain]`` based on rolling accuracy.

    Parameters
    ----------
    state:
        The current environment state (mutated in-place).
    last_correctness:
        Whether the most recent answer was correct.  ``None`` for
        abstain / malformed answers (counted as incorrect).
    domain:
        Override for the active domain.  Defaults to ``state.current_domain``.

    Side-effects
    ------------
    * Appends a new episode record to ``state.episode_history``.
    * May increment or decrement ``state.domain_difficulties[domain]``.
    """
    if domain is None:
        domain = state.current_domain

    current_difficulty = state.domain_difficulties.get(domain, 1)
    global_episode_idx = len(state.episode_history)  # index *before* append

    # Build and append episode record first (so get_rolling_accuracy sees it)
    record: dict = {
        "domain": domain,
        "correct": last_correctness,
        "difficulty": current_difficulty,
        "difficulty_changed": False,
    }
    state.episode_history.append(record)

    # --- compute rolling accuracy after appending ---
    accuracy = get_rolling_accuracy(state, domain)

    # --- hysteresis guard ---
    last_change = _last_change_episode(state, domain)
    episodes_since_change = global_episode_idx - last_change
    if episodes_since_change < HYSTERESIS_EPISODES:
        return  # too soon to change again

    # --- apply threshold rules ---
    new_difficulty = current_difficulty
    if accuracy > HIGH_THRESHOLD:
        new_difficulty = min(current_difficulty + 1, MAX_DIFFICULTY)
    elif accuracy < LOW_THRESHOLD:
        new_difficulty = max(current_difficulty - 1, MIN_DIFFICULTY)

    if new_difficulty != current_difficulty:
        state.domain_difficulties[domain] = new_difficulty
        record["difficulty_changed"] = True  # flag *this* record
