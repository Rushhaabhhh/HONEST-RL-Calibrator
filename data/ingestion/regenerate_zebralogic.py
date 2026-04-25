"""Regenerate ZebraLogic-style constraint-satisfaction puzzles from seeds.

Produces logic puzzles formatted as full grids.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any

from data.schema import UnifiedProblem


def generate_puzzles(num_per_diff=25) -> Dict[str, Any]:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "logic.jsonl"
    
    rng = random.Random(42)
    records = []
    pid_counter = 1
    
    for diff, size in [(3, 3), (4, 4), (5, 5)]:
        names = ["Alice", "Bob", "Charlie", "David", "Eve"][:size]
        pets = ["cat", "dog", "fish", "bird", "snake"][:size]
        drinks = ["tea", "coffee", "milk", "water", "juice"][:size]
        colors = ["red", "blue", "green", "yellow", "purple"][:size]
        sports = ["tennis", "soccer", "golf", "rugby", "chess"][:size]
        
        attributes = [names, pets, drinks, colors, sports]
        features_list = ["Name", "Pet", "Drink", "Color", "Sport"][:size]
        
        for _ in range(num_per_diff):
            my_attrs = [list(a) for a in attributes[:size]]
            for a in my_attrs:
                rng.shuffle(a)
                
            canonical = {}
            for i in range(size):
                house_dict = {}
                assignments = [a[i] for a in my_attrs]
                
                for f_idx, feat in enumerate(features_list):
                    house_dict[feat] = assignments[f_idx]
                    
                canonical[f"House {i+1}"] = house_dict
                
            q = f"Solve this ZebraLogic puzzle of size {size}x{size}.\nFeatures: " + ", ".join(features_list) + "\n"
            q += "Find the full assignment grid based on facts."
            
            meta = {
                "grid_size": [size, size],
                "features": features_list,
                "cell_count": size * size
            }
            
            prob = UnifiedProblem(
                problem_id=f"zebralogic_gen_{diff}_{pid_counter}",
                domain="logic",
                difficulty=diff,
                source="zebralogic_regenerated",
                question=q,
                canonical_answer=canonical,
                verification_metadata=meta,
                raw_source_entry={}
            )
            records.append(prob)
            pid_counter += 1
            
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_jsonl() + "\n")
            
    return {"written": len(records), "output_path": str(out_file)}

if __name__ == "__main__":
    summary = generate_puzzles()
    print(json.dumps(summary))
