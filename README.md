# HONEST

An OpenEnv-compliant environment for evaluating honesty in language models.

## Structure

```
server/          # FastAPI server wrapping the environment
server/generators/  # Question/episode generators
models/          # HonestAction, HonestObservation, HonestState dataclasses
client/          # HTTP client for interacting with the server
eval/            # Evaluation scripts
tests/           # Unit and integration tests
```

## Running

```bash
pip install -r requirements.txt
uvicorn server.app:app --reload
```

## Docker

```bash
docker build -t honest-env .
docker run -p 8000:8000 honest-env
```
