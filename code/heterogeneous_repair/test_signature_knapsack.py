"""Independent small-instance tests for the D03 exact selector."""

from __future__ import annotations

import itertools
import random
import unittest

from solve_signature_knapsack import select_signature_knapsack


def brute_force(changed, degrees, costs, budget):
    best_value = -1
    best = ()
    for mask in range(1 << len(degrees)):
        selected = tuple(i for i in range(len(degrees)) if mask & (1 << i))
        if any(not changed[i] or degrees[i] <= 0 for i in selected):
            continue
        if sum(costs[i] for i in selected) > budget:
            continue
        value = sum(degrees[i] for i in selected)
        if value > best_value:
            best_value = value
            best = selected
    return best_value, best


class SignatureKnapsackTests(unittest.TestCase):
    def test_unit_cost_branch(self):
        result = select_signature_knapsack(
            [True, True, True, False], [5, 9, 7, 100], [1, 1, 1, 1], 2
        )
        self.assertEqual(result.selected_indices, (1, 2))
        self.assertEqual(result.removed_worst_case_degree, 16)
        self.assertEqual(result.branch, "unit_cost_degree")

    def test_heterogeneous_counterexample(self):
        result = select_signature_knapsack(
            [True, True, True], [100, 90, 90], [6, 5, 5], 10
        )
        self.assertEqual(result.selected_indices, (1, 2))
        self.assertEqual(result.removed_worst_case_degree, 180)
        self.assertEqual(result.used_cost, 10)

    def test_zero_degree_and_stable_cells_are_excluded(self):
        result = select_signature_knapsack(
            [True, False, True], [0, 100, 4], [1, 1, 2], 3
        )
        self.assertEqual(result.selected_indices, (2,))

    def test_invalid_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            select_signature_knapsack([True], [1], [0], 1)

    def test_random_small_instances_match_brute_force(self):
        rng = random.Random(20261112)
        for _ in range(1000):
            n = rng.randint(1, 8)
            changed = [rng.random() < 0.8 for _ in range(n)]
            degrees = [rng.randint(0, 20) for _ in range(n)]
            costs = [rng.randint(1, 8) for _ in range(n)]
            budget = rng.randint(0, sum(costs))
            expected, _ = brute_force(changed, degrees, costs, budget)
            result = select_signature_knapsack(changed, degrees, costs, budget)
            self.assertEqual(result.removed_worst_case_degree, expected)
            self.assertLessEqual(result.used_cost, budget)
            self.assertTrue(all(changed[i] and degrees[i] > 0 for i in result.selected_indices))


if __name__ == "__main__":
    unittest.main()
