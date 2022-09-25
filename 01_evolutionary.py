import copy
import operator
import random
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import cauchy

# HYPERPARAMETERS
# Meta-EP
gens_meta = 50
mu_meta = 100
# beta = (30 - (-30)) / 10
# gamma = 0
# epsilon = (30-(-30))/10
# var_control = 1
# Diff ev
mu_de = 50


def generate_standard_normal_vector(length):
    return np.array([random.normalvariate(0, 1) for _ in range(length)])


def generate_cauchy_vector(length):
    return cauchy.rvs(size=length)


def generate_uniform_vector(length, xmin, xmax):
    return np.array([random.uniform(xmin, xmax) for _ in range(length)])


def min_max_normalise(value, newMin, newMax, oldMin, oldMax):
    return (value - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin


class EPIndividual:
    def __init__(self, n, evaluator, xmin=-1, xmax=1):
        self.xmin = xmin
        self.xmax = xmax
        self.vector = generate_uniform_vector(n, xmin, xmax)
        self.variance = np.array([3.0 for _ in range(n)])
        self.evaluator = evaluator
        self.fitness = None
        self.f = None

    def __lt__(self, other):
        if self.fitness is None:
            self.evaluate()
        if other.fitness is None:
            other.evaluate()

        return self.fitness < other.fitness

    def __eq__(self, other):
        if self.fitness is None:
            self.evaluate()
        if other.fitness is None:
            other.evaluate()

        return self.fitness == other.fitness

    def __repr__(self):
        return '{:,}'.format(round(self.fitness, 4))

    def __hash__(self):
        return hash(tuple(self.vector) + tuple(self.variance))

    def mutate(self):
        if self.fitness is None:
            self.evaluate()

        offspring = copy.deepcopy(self)

        n = len(self.vector)
        offspring.vector += offspring.variance * generate_cauchy_vector(n)  # delta_i times eta_i

        tau = 1 / np.sqrt(2 * (np.sqrt(n)))
        tau_prime = 1 / np.sqrt(2 * n)
        offspring.variance *= np.exp(tau_prime * random.normalvariate(0, 1) +
                                     tau * generate_standard_normal_vector(n))

        # variance_increment = generate_standard_normal_vector(len(self.variance))  # r_vi
        # variance_increment *= np.sqrt(abs(self.variance * var_control))  # times sqrt(c*v_i)
        # offspring.variance += variance_increment
        # offspring.variance = np.maximum(variance_increment, epsilon)

        offspring.evaluate()
        if np.isinf(offspring.fitness):
            return self.mutate()

        return offspring

    def evaluate(self):
        if any(x < self.xmin or x > self.xmax for x in self.vector):
            self.fitness = np.inf
            return

        self.fitness = self.evaluator(self)


def rosenbrock(individual):
    fitness = 0
    for d in range(0, len(individual.vector) - 1):
        fitness += 100 * (individual.vector[d] ** 2 - individual.vector[d + 1]) ** 2 + (
                individual.vector[d] - 1) ** 2

    return fitness


def griewank(individual):
    summation = 0
    product = 1
    for d in range(0, len(individual.vector)):
        summation += individual.vector[d] ** 2
        product *= np.cos(individual.vector[d] / np.sqrt(d + 1))

    return summation / 4000 - product + 1


def basicEP(dimensions, fitness):
    population = [EPIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu_meta)]
    for individual in population:
        individual.evaluate()
    for _ in range(gens_meta):
        offspring = []
        for individual in population:
            child = individual.mutate()
            offspring.append(child)

        # Tournament time
        tournament_pool = population + offspring
        q = int(mu_meta * 0.1)
        for individual in tournament_pool:
            opponents = random.sample(set(tournament_pool) - {individual}, q)
            individual.wins = reduce(operator.add,
                                     map(lambda _: 1,
                                         filter(lambda x: x.fitness > individual.fitness,
                                                opponents)),
                                     0)

        population = sorted(tournament_pool, key=lambda x: x.wins, reverse=True)[0:mu_meta]
        population = sorted(population)

    return population[0].fitness


def differential_evolution(dimensions, fitness):
    population = [EPIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu_de)]


def repeat(n, dimensions, fitnesses):
    for fitness in fitnesses:
        print("## " + fitness.__name__.title())
        print("### Basic EP")
        stats_ep = pd.DataFrame(columns=["Gen", "Best"])
        for i in range(n):
            best = basicEP(dimensions, fitness)
            stats_ep.loc[i] = [i, best]
        print(stats_ep["Best"].describe())
        print("### Differential evolution")
        # for _ in range(n):
        #     differential_evolution(dimensions, fitness)


def main(seed=None):
    repeats = 30

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Rosenbrock and Griewank function at D = 20
    print("# D = 20")
    repeat(repeats, 20, [rosenbrock, griewank])

    print("# D = 50")
    # Rosenbrock function at D = 50
    repeat(repeats, 50, [rosenbrock])


if __name__ == "__main__":
    main(seed=0)
