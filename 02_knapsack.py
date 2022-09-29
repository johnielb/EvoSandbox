import numpy as np
import pandas as pd
from numpy.random import choice
import sys
import random
from functools import cmp_to_key

# HYPERPARAMETERS
mu = 100
p_cross = 1
p_mutate = 0.25
n_elite = 2
epochs = 500

verbose = False
global candidates, capacity


def parse_data_file(fname):
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


def generate_population(length):
    population = []
    for i in range(mu):
        ind = []
        for j in range(length):
            ind.append(random.choice([True, False]))
        population.append(ind)

    return population


def evaluate(individual):
    alpha = 2
    value = 0
    weight = 0
    for n, item in enumerate(individual):
        if item:
            tuple = candidates[n]
            value += tuple[0]
            weight += tuple[1]

    penalty = alpha * max(0, weight - capacity)
    return value - penalty


def evaluate_value(individual):
    value = 0
    for n, item in enumerate(individual):
        if item:
            tuple = candidates[n]
            value += tuple[0]
    return value


def is_feasible(individual):
    weight = 0
    for n, item in enumerate(individual):
        if item:
            tuple = candidates[n]
            weight += tuple[1]
    return weight <= capacity


def sort_population(i1, i2):
    return evaluate(i2) - evaluate(i1)


def roulette_probabilities(population):
    values = [evaluate(i) for i in population]
    vmin, vmax = min(values)-1, max(values)  # -1 to allow the minimum value to occupy a non-zero slice
    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return [x / sum(norm) for x in norm]


def one_point_crossover(i1, i2):
    ind_len = len(i1)
    idx = random.choice(range(1, ind_len-1))
    c1 = i1[0:idx] + i2[idx:ind_len]
    c2 = i2[0:idx] + i1[idx:ind_len]
    return c1, c2


def mutate(individual):
    idx = random.choice(range(len(individual)))
    individual[idx] = not individual[idx]
    return individual


def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    for f in range(1, len(sys.argv)):  # start from 1, skip 0th argument - filename
        global candidates, capacity
        candidates, capacity = parse_data_file(sys.argv[f])
        n = len(candidates)

        # Randomly initialise a population of individuals (bit string, each bit has
        # 50% probability to be 1, and 50% to be 0)
        population = generate_population(n)

        solution = None
        convergence = []
        for epoch in range(epochs):
            # Fitness evaluation of each individual
            population = sorted(population, key=cmp_to_key(sort_population))
            feasible_population = [ind for ind in population if is_feasible(ind)]

            # Update the best feasible solution
            candidate_solution = feasible_population[0] if len(feasible_population) > 0 else population[0]
            # Only if it's better than the current solution
            if solution is None or evaluate(candidate_solution) > evaluate(solution):
                solution = candidate_solution

            # Do elitism (copy top individuals to the new generation)
            children = feasible_population[0:n_elite] if len(feasible_population) >= n_elite else population[0:n_elite]

            # Sample top 5 for average fitness
            average_fitness = np.mean([evaluate(ind) for ind in population[0:5]])
            convergence.append(average_fitness)

            # Generate probabilities for roulette wheel selection
            roulette = roulette_probabilities(population)

            # Repeat until the new population is full:
            while len(children) < mu:
                parents = choice(range(len(roulette)), p=roulette, size=2, replace=False)
                child1, child2 = population[parents[0]], population[parents[1]]
                if random.random() < p_cross:
                    child1, child2 = one_point_crossover(child1, child2)
                if random.random() < p_mutate:
                    child1 = mutate(child1)
                if random.random() < p_mutate:
                    child2 = mutate(child2)
                children += [child1, child2]
            population = children

        print("Solution:", solution)
        print("Value:", evaluate(solution))
        filename = sys.argv[f].split("/")[-1]
        pd.DataFrame(convergence).to_csv("out/part1/"+filename+"_"+str(seed)+".csv")


if __name__ == '__main__':
    for i in range(5):
        print("Seed =", i)
        main(i)
