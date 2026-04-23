import random
from typing import Callable, List, Optional, Tuple

def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str]:
    """Generate a Python code execution problem."""
    rng = random.Random(seed) if seed is not None else random
    if difficulty == 1:
        return _level_1(rng)
    if difficulty == 2:
        return _level_2(rng)
    if difficulty == 3:
        return _level_3(rng)
    if difficulty == 4:
        return _level_4(rng)
    if difficulty == 5:
        return _level_5(rng)
    raise ValueError(f"difficulty must be in 1..5, got {difficulty}")

def _run_and_format(code: str, val: int) -> Tuple[str, str]:
    local_env = {}
    exec(code, local_env, local_env)
    f = local_env['f']
    result = f(val)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    ans = str(result)
    q = f"Given this Python function:\n```python\n{code}\n```\nwhat does f({val}) return?"
    return q, ans

def _level_1(rng) -> Tuple[str, str]:
    val = rng.randint(1, 20)
    op = rng.choice(["+", "-", "*"])
    c = rng.randint(1, 20)
    d = rng.randint(1, 20)
    code = f"def f(x):\n    return (x {op} {c}) + {d}"
    return _run_and_format(code, val)

def _level_2(rng) -> Tuple[str, str]:
    val = rng.randint(1, 20)
    c = rng.randint(5, 15)
    op = rng.choice([">", "<", ">=", "<=", "=="])
    ret1 = rng.randint(1, 100)
    ret2 = rng.randint(1, 100)
    code = f"def f(x):\n    if x {op} {c}:\n        return {ret1}\n    else:\n        return {ret2}"
    return _run_and_format(code, val)

def _level_3(rng) -> Tuple[str, str]:
    val = rng.randint(2, 5)
    start = rng.randint(1, 10)
    op = rng.choice(["+", "*"])
    if op == "+":
        code = f"def f(x):\n    total = {start}\n    for i in range(x):\n        total += i\n    return total"
    else:
        code = f"def f(x):\n    total = {start}\n    for i in range(x):\n        total = total * {rng.randint(2, 3)}\n    return total"
    return _run_and_format(code, val)

def _level_4(rng) -> Tuple[str, str]:
    val = rng.randint(3, 6)
    c = rng.randint(2, 5)
    code = f"def f(x):\n    total = 0\n    for i in range(1, x + 1):\n        if i % 2 == 0:\n            total += {c}\n        else:\n            total *= 2\n    return total"
    return _run_and_format(code, val)

def _level_5(rng) -> Tuple[str, str]:
    val = rng.randint(3, 6)
    base = rng.randint(1, 5)
    c = rng.randint(2, 5)
    code1 = f"def f(x):\n    if x <= 0:\n        return {base}\n    return {c} + f(x - 1)"
    code2 = f"def f(x):\n    if x <= 0:\n        return {base}\n    return x * f(x - 1)"
    code = rng.choice([code1, code2])
    return _run_and_format(code, val)
