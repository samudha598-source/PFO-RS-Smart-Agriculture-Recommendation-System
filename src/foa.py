# src/foa.py
# Permutation Flamingo Optimization (PFO) for feature selection

import numpy as np
import random


class PermutationFlamingoOptimizer:
    """
    Permutation Flamingo Optimization (PFO) algorithm used for feature selection.

    Each individual represents a permutation of feature indices.
    The first k indices of the permutation represent the selected feature subset.
    """

    def __init__(self, num_features, fitness_fn, cfg):
        self.num_features = num_features
        self.fitness_fn = fitness_fn

        self.population_size = cfg["pfo"]["population_size"]
        self.iterations = cfg["pfo"]["iterations"]

        self.min_k = cfg["pfo"]["topk"]["min_k"]
        self.max_k = cfg["pfo"]["topk"]["max_k"]

        self.lambda_size = cfg["pfo"]["lambda_size"]

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            perm = np.random.permutation(self.num_features)
            population.append(perm)
        return population

    def _decode_subset(self, perm):
        """
        Select top-k features from permutation.
        """
        k = random.randint(self.min_k, self.max_k)
        return perm[:k], k

    def _mutation(self, perm):
        """
        Simple swap mutation.
        """
        a, b = random.sample(range(self.num_features), 2)
        perm[a], perm[b] = perm[b], perm[a]
        return perm

    def _local_search(self, perm):
        """
        Local permutation refinement using random swaps.
        """
        new_perm = perm.copy()
        for _ in range(2):
            i, j = random.sample(range(self.num_features), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Main optimization loop.
        """
        population = self._initialize_population()

        best_perm = None
        best_score = float("inf")

        for iteration in range(self.iterations):

            new_population = []

            for perm in population:

                subset, k = self._decode_subset(perm)

                score = self.fitness_fn(
                    X_train[:, subset],
                    y_train,
                    X_val[:, subset],
                    y_val,
                )

                penalty = self.lambda_size * (k / self.num_features)

                fitness = score + penalty

                if fitness < best_score:
                    best_score = fitness
                    best_perm = perm.copy()

                # Flamingo-inspired update
                candidate = self._mutation(perm.copy())

                if random.random() < 0.3:
                    candidate = self._local_search(candidate)

                new_population.append(candidate)

            population = new_population

            print(f"FOA Iteration {iteration+1}/{self.iterations} Best Fitness: {best_score:.4f}")

        best_subset, _ = self._decode_subset(best_perm)

        return best_subset.tolist()
