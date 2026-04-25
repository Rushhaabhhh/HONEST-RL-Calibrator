from server.environment import HonestEnvironment
from models.models import HonestAction

env = HonestEnvironment()
obs = env.reset(seed=42)
print("Initial obs:", obs.question[:50], obs.domain, obs.difficulty)

action = HonestAction(raw_text="<reasoning>thinking...</reasoning><answer>1</answer><analysis>critique</analysis><confidence>1.0</confidence>")
obs = env.step(action)
print("Next obs:", obs.question[:50], obs.domain, obs.difficulty, obs.reward)
