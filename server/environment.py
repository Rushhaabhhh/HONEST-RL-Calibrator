"""HonestEnvironment — main environment class for the HONEST calibration benchmark."""

import logging
import random
import uuid
from typing import Any, Optional

from models.models import HonestAction, HonestObservation, HonestState
from openenv.core.env_server.interfaces import Environment
from data.sampler.unified_sampler import generate_code, generate_logic, generate_math
from server.difficulty import update_difficulty
from server.reward import compute_reward, parse_action

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "logic"]
EPISODE_LENGTH = 5
INITIAL_DIFFICULTIES = {"math": 1, "code": 1, "logic": 1}


class HonestEnvironment(Environment):
    """HONEST: Honesty-Optimised and Normalized Environment for Self-Triage.

    Each episode presents the agent with a sequence of questions drawn from
    three domains (math, code, logic) at adaptively-chosen difficulty levels.
    The agent must respond with an <answer>/<confidence> pair or <abstain/>.
    Rewards are computed using the Brier-score calibration scheme.
    """

    # All mutable state lives inside self._state — no class-level shared state.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state: HonestState = HonestState(episode_id="")
        self._generators = {
            "math": generate_math,
            "code": generate_code,
            "logic": generate_logic,
        }
        self._current_question: Optional[str] = None
        self._current_answer: Optional[str] = None
        self._current_problem_id: Optional[str] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HonestObservation:
        """Start a new episode and return the first observation."""
        ep_id = episode_id or str(uuid.uuid4())

        self._state = HonestState(
            episode_id=ep_id,
            domain_difficulties=dict(INITIAL_DIFFICULTIES),
            episode_step=0,
            episode_history=[],
        )

        rng = random.Random(seed) if seed is not None else random
        domain = rng.choice(DOMAINS)
        self._state.current_domain = domain

        difficulty = self._state.domain_difficulties[domain]
        question, answer, problem_id = self._generators[domain](difficulty, seed=seed)
        self._current_question = question
        self._current_answer = answer
        self._current_problem_id = problem_id

        logger.info(
            "reset: episode_id=%s domain=%s difficulty=%d",
            ep_id,
            domain,
            difficulty,
        )

        return HonestObservation(
            question=question,
            domain=domain,
            difficulty=difficulty,
            episode_step=0,
            done=False,
            reward=None,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: HonestAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HonestObservation:
        """Process one agent action and advance the environment."""
        domain = self._state.current_domain
        difficulty = self._state.domain_difficulties[domain]

        parsed = parse_action(action.raw_text)
        reward_value, correctness = compute_reward(
            parsed,
            self._current_answer,
            difficulty,
            problem_id=self._current_problem_id,
            domain=domain,
        )

        # Record episode history
        self._state.episode_history.append(
            {
                "question": self._current_question,
                "ground_truth": self._current_answer,
                "parsed": parsed,
                "correct": correctness,
                "reward": reward_value,
                "domain": domain,
                "difficulty": difficulty,
            }
        )

        self._state.episode_step += 1
        update_difficulty(self._state, correctness, domain=domain)

        terminal = self._state.episode_step >= EPISODE_LENGTH

        logger.info(
            "step %d: domain=%s difficulty=%d parsed_type=%s reward=%.4f correct=%s terminal=%s",
            self._state.episode_step,
            domain,
            difficulty,
            parsed.get("type"),
            reward_value,
            correctness,
            terminal,
        )

        if terminal:
            return HonestObservation(
                question="",
                domain=domain,
                difficulty=difficulty,
                episode_step=self._state.episode_step,
                previous_correctness=correctness,
                terminal=True,
                done=True,
                reward=reward_value,
            )

        # Pick next problem
        next_domain = random.choice(DOMAINS)
        self._state.current_domain = next_domain
        next_difficulty = self._state.domain_difficulties[next_domain]
        next_question, next_answer, next_problem_id = self._generators[next_domain](next_difficulty)
        self._current_question = next_question
        self._current_answer = next_answer
        self._current_problem_id = next_problem_id

        return HonestObservation(
            question=next_question,
            domain=next_domain,
            difficulty=next_difficulty,
            episode_step=self._state.episode_step,
            previous_correctness=correctness,
            terminal=False,
            done=False,
            reward=reward_value,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> HonestState:
        return self._state
