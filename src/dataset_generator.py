import random
import itertools
from typing import List, Dict, Any

def solve_countdown_all(numbers: List[int]) -> Dict[int, str]:
    """Trova tutti i target interi positivi raggiungibili con i numeri dati."""
    targets = {}

    for length in range(2, len(numbers) + 1):
        for subset in itertools.permutations(numbers, length):
            def eval_tree(items):
                if len(items) == 1:
                    return [(items[0], str(items[0]))]
                results = []
                for split in range(1, len(items)):
                    lefts = eval_tree(items[:split])
                    rights = eval_tree(items[split:])
                    for lv, le in lefts:
                        for rv, re in rights:
                            results.append((lv + rv, f"({le} + {re})"))
                            if lv - rv > 0:
                                results.append((lv - rv, f"({le} - {re})"))
                            results.append((lv * rv, f"({le} * {re})"))
                            if rv > 0 and lv % rv == 0:
                                results.append((lv // rv, f"({le} / {re})"))
                return results

            for v, e in eval_tree(list(subset)):
                if 1 <= v <= 200:
                    targets[v] = e

    return targets

def generate_countdown_dataset(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Genera un dataset di puzzle Countdown garantiti solubili."""
    random.seed(seed)
    dataset = []
    seen = set()

    attempts = 0
    while len(dataset) < num_samples and attempts < num_samples * 25:
        attempts += 1
        numbers = [random.randint(1, 100) for _ in range(4)]
        key = tuple(sorted(numbers))
        if key in seen:
            continue

        reachable = solve_countdown_all(numbers)
        valid_targets = [t for t in reachable.keys() if t not in numbers and t >= 10]

        if valid_targets:
            chosen_target = random.choice(valid_targets)
            sample = {
                "id": len(dataset) + 1,
                "numbers": numbers,
                "target": chosen_target,
                "example_solution": reachable[chosen_target],
                "prompt": f"Using the numbers {numbers}, create an equation that equals {chosen_target}. You can use basic arithmetic operations (+, -, *, /) and each number can be used at most once. Show your reasoning in <think>...</think> and place the final equation in <answer>...</answer>."
            }
            dataset.append(sample)
            seen.add(key)

    return dataset
