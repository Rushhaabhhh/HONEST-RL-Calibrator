"""HonestEnv — OpenEnv-compatible async client for the HONEST environment server."""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from models.models import HonestAction, HonestObservation, HonestState


class HonestEnv(EnvClient[HonestAction, HonestObservation, HonestState]):
    """Async WebSocket client for the HONEST calibration environment.

    Usage (async)::

        async with HonestEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            print(result.observation.question)

            action = HonestAction(raw_text="<answer>42</answer><confidence>0.8</confidence>")
            result = await env.step(action)
            print(result.reward)

    Usage (sync wrapper)::

        client = HonestEnv(base_url="http://localhost:8000").sync()
        with client:
            result = client.reset()
            result = client.step(HonestAction(raw_text="<abstain/>"))
    """

    # ------------------------------------------------------------------
    # Required abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: HonestAction) -> Dict[str, Any]:
        """Serialize HonestAction → JSON dict for the /step wire format."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[HonestObservation]:
        """Deserialize server response → StepResult[HonestObservation]."""
        obs_data = payload.get("observation", {})
        observation = HonestObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> HonestState:
        """Deserialize state response → HonestState."""
        return HonestState(**payload)

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    async def query(self) -> Dict[str, Any]:
        """Reset the environment and return a summary of the first question.

        Returns::

            {
                "question":   str,   # the full question text
                "domain":     str,   # "math" | "code" | "logic"
                "difficulty": int,   # 1–5
            }
        """
        result = await self.reset()
        obs = result.observation
        return {
            "question": obs.question,
            "domain": obs.domain,
            "difficulty": obs.difficulty,
        }
