from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class HonestAction(Action):
    raw_text: str


class HonestObservation(Observation):
    question: str
    domain: str
    difficulty: int
    episode_step: int
    previous_correctness: Optional[bool] = None
    revealed_answer: Optional[str] = None
    # terminal mirrors Observation.done; kept separate for semantic clarity
    terminal: bool = False


class HonestState(State):
    current_domain: str = ""
    domain_difficulties: Dict[str, int] = Field(default_factory=dict)
    episode_step: int = 0
    episode_history: List[Any] = Field(default_factory=list)
    hints_revealed: int = 0
