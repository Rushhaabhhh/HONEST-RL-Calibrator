from openenv.core.env_server.interfaces import Environment

from models.models import HonestAction, HonestObservation, HonestState


class HonestEnvironment(Environment):
    """HONEST: environment logic goes here."""

    def reset(self) -> HonestObservation:
        raise NotImplementedError

    def step(self, action: HonestAction) -> HonestObservation:
        raise NotImplementedError

    @property
    def state(self) -> HonestState:
        raise NotImplementedError
