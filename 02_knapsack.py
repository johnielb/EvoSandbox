import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import choice
import sys
import random
from functools import cmp_to_key

# HYPERPARAMETERS
mu = 500
M = int(mu*0.25)
n_elitism = int(mu*0.05)
epochs = 25

verbose = False
global candidates, capacity


def parse_data_file(fname):
    if verbose:
        print("===", fname, "===")

    file = open(fname)
    header = file.readline().split(" ")
    n = int(header[0])
    capacity = int(header[1])
    if verbose:
        print(n, "items, capacity", capacity)

    candidates = {}
    for i in range(n):
        line = file.readline().split(" ")
        # Append tuple to candidates, giving it a randomly accessible name i
        candidates[i] = (int(line[0]), int(line[1]))

    assert len(candidates) == n
    return candidates, capacity


def evaluate(individual):
    alpha = 50
    value = 0
    weight = 0
    for n, item in enumerate(individual):
        if item:
            candidate = candidates[n]
            value += candidate[0]
            weight += candidate[1]

    penalty = alpha * max(0, weight - capacity)
    return value - penalty


def sort_population(i1, i2):
    return evaluate(i2) - evaluate(i1)


def main(fname):
    convergence = []

    global candidates, capacity
    candidates, capacity = parse_data_file(fname)
    n = len(candidates)
    solutions = []

    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        if verbose:
            print("Seed =", seed)

        # Randomly initialise a population of individuals (bit string, each bit has
        # 50% probability to be 1, and 50% to be 0)
        population = [[random.choice([True, False]) for _ in range(n)] for _ in range(mu)]
        population = sorted(population, key=cmp_to_key(sort_population))

        seed_convergence = []
        for epoch in range(epochs):
            # Fitness evaluation of each individual, get best M
            parents = population[0:M]
            elites = population[0:n_elitism]

            # For each bit, calculate probability
            probability = [sum([ind[k] for ind in parents]) / M for k in range(n)]

            # Construct individual based on probability
            population = elites + [[True if random.uniform(0, 1) < p else False for p in probability] for _ in range(mu-n_elitism)]
            population = sorted(population, key=cmp_to_key(sort_population))

            # Record stats
            best = population[0:5]
            average = np.mean([evaluate(ind) for ind in best])
            seed_convergence.append(average)

        solution = sorted(population, key=cmp_to_key(sort_population))[0]

        if verbose:
            print("Solution:", solution)
            print("Value:", evaluate(solution))
        solutions.append(evaluate(solution))
        convergence.append(seed_convergence)

    print(solutions)
    print("Mean: %.2f" % np.mean(solutions))
    print("SD: %.2f" % np.std(solutions))

    return pd.DataFrame(convergence).T


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for f in range(1, len(sys.argv)):  # start from 1, skip 0th argument - script filename
        df = main(sys.argv[f])
        ax[f-1].set_title(sys.argv[f])
        df.plot(ax=ax[f-1])
        if df.min().min() < 0:
            ax[f-1].set_ylim([0, df.max().max()+100])
    fig.supxlabel("Epoch")
    fig.supylabel("Fitness (value)")
    fig.show()
