from openenv.core.env_server.http_server import create_app

from .environment import HonestEnvironment

environment = HonestEnvironment()
app = create_app(environment)
