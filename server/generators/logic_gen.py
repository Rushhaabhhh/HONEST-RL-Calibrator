import random
from typing import Optional, Tuple
from constraint import Problem, AllDifferentConstraint

def generate(difficulty: int, seed: Optional[int] = None) -> Tuple[str, str]:
    """Generate a logic deduction problem."""
    rng = random.Random(seed) if seed is not None else random
    
    for _ in range(100):
        if difficulty == 1:
            res = _level_1(rng)
        elif difficulty == 2:
            res = _level_2(rng)
        elif difficulty == 3:
            res = _level_3(rng)
        elif difficulty == 4:
            res = _level_4(rng)
        elif difficulty == 5:
            res = _level_5(rng)
        else:
            raise ValueError(f"difficulty must be in 1..5, got {difficulty}")
            
        if res is not None:
            return res
            
    raise RuntimeError("Failed to generate a puzzle with exactly 1 unique solution.")

def _level_1(rng) -> Optional[Tuple[str, str]]:
    names = rng.sample(["A", "B"], 2)
    p = Problem()
    p.addVariables(names, [1, 2])
    p.addConstraint(AllDifferentConstraint())
    
    n1, n2 = names
    p.addConstraint(lambda a, b: a > b, (n1, n2))
    
    sols = p.getSolutions()
    if len(sols) == 1:
        q = f"There are 2 entities: {n1} and {n2}.\n{n1} is strictly greater than {n2}.\nWho is the largest?"
        return q, n1
    return None

def _level_2(rng) -> Optional[Tuple[str, str]]:
    num = 4
    names = rng.sample(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], num)
    
    # Generate a ground truth
    ground_truth = {name: rank for name, rank in zip(names, rng.sample(range(1, num + 1), num))}
    
    rules = []
    
    # We will incrementally add valid constraints until exactly 1 solution
    p = Problem()
    p.addVariables(names, list(range(1, num + 1)))
    p.addConstraint(AllDifferentConstraint())
    
    # Pre-generate valid constraint candidates
    candidates = []
    for n1 in names:
        for n2 in names:
            if n1 != n2:
                if ground_truth[n1] < ground_truth[n2]:
                    candidates.append((n1, n2, "before"))
                else:
                    candidates.append((n1, n2, "after"))
                    
    rng.shuffle(candidates)
    
    for (n1, n2, rtype) in candidates:
        if rtype == "before":
            p.addConstraint(lambda a, b: a < b, (n1, n2))
            rules.append(f"{n1} is ranked before {n2}.")
        else:
            p.addConstraint(lambda a, b: a > b, (n1, n2))
            rules.append(f"{n1} is ranked after {n2}.")
            
        sols = p.getSolutions()
        if len(sols) == 0:
            return None # Should not happen unless our logic is wrong
        if len(sols) == 1:
            first = [k for k, v in sols[0].items() if v == 1][0]
            rng.shuffle(rules)
            q = f"There are {num} entities ranked 1 to {num}: " + ", ".join(names) + ".\n" + "\n".join(rules) + "\nWho is ranked 1st?"
            return q, first
            
    return None

def _level_3(rng) -> Optional[Tuple[str, str]]:
    num = 4
    names = rng.sample(["Olivia", "Liam", "Emma", "Noah", "Ava", "Oliver"], num)
    ground_truth = {name: rank for name, rank in zip(names, rng.sample(range(1, num + 1), num))}
    
    rules = []
    p = Problem()
    p.addVariables(names, list(range(1, num + 1)))
    p.addConstraint(AllDifferentConstraint())
    
    candidates = []
    for n1 in names:
        for n2 in names:
            if n1 != n2:
                if abs(ground_truth[n1] - ground_truth[n2]) != 1:
                    candidates.append((n1, n2, "not_next"))
                if ground_truth[n1] < ground_truth[n2]:
                    candidates.append((n1, n2, "before"))
        # Not specific position
        for pos in range(1, num + 1):
            if ground_truth[n1] != pos:
                candidates.append((n1, pos, "not_specific"))
                
    rng.shuffle(candidates)
    
    unique_rules = set()

    for item in candidates:
        rtype = item[2]
        if rtype == "not_next":
            n1, n2 = item[0], item[1]
            key = tuple(sorted([n1, n2])) + ("not_next",)
            if key in unique_rules: continue
            unique_rules.add(key)
            # local binding workaround
            def make_cons_not_next():
                return lambda a, b: abs(a - b) != 1
            p.addConstraint(make_cons_not_next(), (n1, n2))
            rules.append(f"{n1} is not ranked immediately next to {n2}.")
            
        elif rtype == "not_specific":
            n1, pos = item[0], item[1]
            key = (n1, pos, "not_specific")
            if key in unique_rules: continue
            unique_rules.add(key)
            def make_cons_not_pos(v):
                return lambda a: a != v
            p.addConstraint(make_cons_not_pos(pos), (n1,))
            rules.append(f"{n1} is not ranked {pos}.")
            
        elif rtype == "before":
            n1, n2 = item[0], item[1]
            key = (n1, n2, "before")
            if key in unique_rules: continue
            unique_rules.add(key)
            def make_cons_before():
                return lambda a, b: a < b
            p.addConstraint(make_cons_before(), (n1, n2))
            rules.append(f"{n1} is ranked before {n2}.")
            
        sols = p.getSolutions()
        if len(sols) == 1:
            last = [k for k, v in sols[0].items() if v == num][0]
            rng.shuffle(rules)
            q = f"There are {num} entities ranked 1 to {num}: " + ", ".join(names) + ".\n" + "\n".join(rules) + f"\nWho is ranked {num}?"
            return q, last
            
    return None

def _level_4(rng) -> Optional[Tuple[str, str]]:
    names = rng.sample(["Alice", "Bob", "Charlie", "David"], 3)
    colors = rng.sample(["Red", "Blue", "Green", "Yellow"], 3)
    ages = rng.sample([20, 25, 30, 35], 3)
    
    # Ground truth mapping to positions 1, 2, 3
    gt_n = dict(zip([f"N_{n}" for n in names], [1, 2, 3]))
    gt_c = dict(zip([f"C_{c}" for c in colors], rng.sample([1, 2, 3], 3)))
    gt_a = dict(zip([f"A_{a}" for a in ages], rng.sample([1, 2, 3], 3)))
    
    ground_truth = {**gt_n, **gt_c, **gt_a}
    
    p = Problem()
    for name_var, pos in gt_n.items():
        p.addVariable(name_var, [pos])
    p.addVariables(list(gt_c.keys()), [1, 2, 3])
    p.addVariables(list(gt_a.keys()), [1, 2, 3])
    p.addConstraint(AllDifferentConstraint(), list(gt_n.keys()))
    p.addConstraint(AllDifferentConstraint(), list(gt_c.keys()))
    p.addConstraint(AllDifferentConstraint(), list(gt_a.keys()))
    
    all_vars = list(ground_truth.keys())
    candidates = []
    
    for v1 in all_vars:
        for v2 in all_vars:
            if v1[0] != v2[0] and v1 != v2:
                if ground_truth[v1] == ground_truth[v2]:
                    candidates.append((v1, v2, "is"))
                else:
                    candidates.append((v1, v2, "is_not"))
                    
    rng.shuffle(candidates)
    
    def get_printable(v):
        if v.startswith("N_"): return f"the person named {v[2:]}"
        if v.startswith("C_"): return f"the person who likes {v[2:]}"
        return f"the {v[2:]}-year-old"

    rules = []
    added_pairs = set()

    for v1, v2, rel in candidates:
        # avoid symmetric duplicate rules
        pair_key = tuple(sorted([v1, v2]))
        if pair_key in added_pairs:
            continue
        added_pairs.add(pair_key)
        
        if rel == "is":
            def make_is(): return lambda a, b: a == b
            p.addConstraint(make_is(), (v1, v2))
            rules.append(f"{get_printable(v1).capitalize()} is {get_printable(v2)}.")
        else:
            def make_is_not(): return lambda a, b: a != b
            p.addConstraint(make_is_not(), (v1, v2))
            rules.append(f"{get_printable(v1).capitalize()} is not {get_printable(v2)}.")
            
        sols = p.getSolutions()
        if len(sols) == 1:
            n_ask = names[0]
            n_pos = sols[0][f"N_{n_ask}"]
            c_ans = [c for c in colors if sols[0][f"C_{c}"] == n_pos][0]
            
            rng.shuffle(rules)
            q = (f"There are 3 people. They have unique names, ages, and favorite colors.\n"
                 f"Names: " + ", ".join(names) + "\n"
                 f"Colors: " + ", ".join(colors) + "\n"
                 f"Ages: " + ", ".join(map(str, ages)) + "\n\n"
                 f"Facts:\n- " + "\n- ".join(rules) + f"\n\nBased ONLY on these facts, what color does {n_ask} like? Provide exactly the color string.")
            return q, c_ans
            
    return None

def _level_5(rng) -> Optional[Tuple[str, str]]:
    num = 4
    names = rng.sample(["Alice", "Bob", "Charlie", "David", "Eve"], num)
    colors = rng.sample(["Red", "Blue", "Green", "Yellow", "Purple"], num)
    pets = rng.sample(["Dog", "Cat", "Fish", "Bird", "Snake"], num)
    
    all_n = [f"N_{n}" for n in names]
    all_c = [f"C_{c}" for c in colors]
    all_p = [f"P_{pt}" for pt in pets]
    
    gt = {}
    gt.update(dict(zip(all_n, range(1, num + 1))))
    for lst in [all_c, all_p]:
        gt.update(dict(zip(lst, rng.sample(range(1, num + 1), num))))
        
    p = Problem()
    for name_var in all_n:
        p.addVariable(name_var, [gt[name_var]])
        
    for lst in [all_c, all_p]:
        p.addVariables(lst, list(range(1, num + 1)))
        p.addConstraint(AllDifferentConstraint(), lst)
        
    all_vars = list(gt.keys())
    candidates = []
    
    for v1 in all_vars:
        for v2 in all_vars:
            if v1[0] != v2[0] and v1 != v2:
                if gt[v1] == gt[v2]:
                    candidates.append((v1, v2, "is"))
                elif abs(gt[v1] - gt[v2]) == 1:
                    candidates.append((v1, v2, "next_to"))
                elif gt[v1] == gt[v2] - 1:
                    candidates.append((v1, v2, "left_of"))
                    
    rng.shuffle(candidates)
    
    def get_printable(v):
        if v.startswith("N_"): return f"named {v[2:]}"
        if v.startswith("C_"): return f"in the {v[2:]} shirt"
        return f"with the {v[2:]}"

    rules = []
    added_pairs = set()

    for v1, v2, rel in candidates:
        pair_key = tuple(sorted([v1, v2]))
        if pair_key in added_pairs and rel in ["is", "next_to"]:
            continue
        added_pairs.add(pair_key)
        
        if rel == "is":
            def make_is(): return lambda a, b: a == b
            p.addConstraint(make_is(), (v1, v2))
            rules.append(f"The person {get_printable(v1)} is exactly the same person {get_printable(v2)}.")
        elif rel == "next_to":
            def make_next(): return lambda a, b: abs(a - b) == 1
            p.addConstraint(make_next(), (v1, v2))
            rules.append(f"The person {get_printable(v1)} is seated immediately next to the person {get_printable(v2)}.")
        elif rel == "left_of":
            def make_left(): return lambda a, b: a == b - 1
            p.addConstraint(make_left(), (v1, v2))
            rules.append(f"The person {get_printable(v1)} is seated immediately to the left of the person {get_printable(v2)}.")
            
        sols = p.getSolutions()
        if len(sols) == 1:
            n_ask = names[-1]
            n_pos = sols[0][f"N_{n_ask}"]
            pet_ans = [pt for pt in pets if sols[0][f"P_{pt}"] == n_pos][0]
            
            rng.shuffle(rules)
            q = (f"Four people sit in a row (positions 1-{num}). They have unique Names, Colors, and Pets.\n"
                 f"Names: " + ", ".join(names) + "\n"
                 f"Colors: " + ", ".join(colors) + "\n"
                 f"Pets: " + ", ".join(pets) + "\n\n"
                 f"Facts:\n- " + "\n- ".join(rules) + f"\n\nBased ONLY on the above facts, what is the Pet of {n_ask}? Provide ONLY the exact string from the Pets list.")
            return q, pet_ans
            
    return None
