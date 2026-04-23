import pytest

from server.generators.logic_gen import generate

@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_returns_valid_tuple(difficulty):
    question, answer = generate(difficulty, seed=42)
    assert isinstance(question, str) and question
    assert isinstance(answer, str) and answer
    assert "There are" in question or "Four people" in question

@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_seeded_is_reproducible(difficulty):
    first = generate(difficulty, seed=123)
    second = generate(difficulty, seed=123)
    assert first == second

@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_different_seeds_vary(difficulty):
    outputs = {generate(difficulty, seed=s) for s in range(10)}
    assert len(outputs) > 1, "seeded generation across different seeds should vary"

def test_unseeded_varies():
    outputs = {generate(3) for _ in range(10)}
    assert len(outputs) > 1, "unseeded generation should vary across calls"

def test_invalid_difficulty_raises():
    with pytest.raises(ValueError):
        generate(0)
    with pytest.raises(ValueError):
        generate(6)
