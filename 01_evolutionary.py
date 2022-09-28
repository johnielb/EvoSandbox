import copy
import operator
import random
import sys
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import cauchy

verbose = False
# HYPERPARAMETERS
gens = 250
mu = lambda d: d * 5
# Fast EP
q = lambda d: int(mu(d) * 0.1)

# Diff ev
F = 0.2
r_crossover = 0.3


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
        self.evaluate()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __repr__(self):
        return '{:,}'.format(round(self.fitness, 4))

    def __hash__(self):
        if self.variance is None:
            return hash(tuple(self.vector))
        return hash(tuple(self.vector) + tuple(self.variance))

    def reproduce(self):
        offspring = None

        for _ in range(20):
            n = len(self.vector)

            offspring = copy.deepcopy(self)
            offspring.vector += offspring.variance * generate_cauchy_vector(n)  # delta_i times eta_i

            tau = 1 / np.sqrt(2 * (np.sqrt(n)))
            tau_prime = 1 / np.sqrt(2 * n)
            offspring.variance *= np.exp(tau_prime * random.normalvariate(0, 1) +
                                         tau * generate_standard_normal_vector(n))

            offspring.evaluate()
            if not np.isinf(offspring.fitness):
                break

        return offspring

    def evaluate(self):
        if any(x < self.xmin or x > self.xmax for x in self.vector):
            self.fitness = np.inf
            return

        self.fitness = self.evaluator(self)


class DEIndividual(EPIndividual):
    def __init__(self, n, evaluator, xmin=-1, xmax=1):
        super(DEIndividual, self).__init__(n, evaluator, xmin, xmax)
        self.variance = None

    def reproduce(self, mate1=None, mate2=None):
        if mate1 is None or mate2 is None:
            raise Exception("Must provide two other individuals for reproduction")

        n = len(self.vector)
        offspring = copy.deepcopy(self)

        # MUTATE --------
        donor = self.vector + F * (mate1.vector - mate2.vector)

        # CROSSOVER -----
        j = random.randint(0, n)
        offspring.vector = np.asarray([donor[i] if r <= r_crossover or i == j else self.vector[i]
                                       for i, r in enumerate(generate_uniform_vector(n, 0, 1))])

        offspring.evaluate()

        return offspring


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


def exclude_from_set(individual, population):
    return list(set(population) - {individual})


def fastEP(dimensions, fitness):
    population = [EPIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu(dimensions))]
    for _ in range(gens):
        offspring = []
        for individual in population:
            child = individual.reproduce()
            offspring.append(child)

        # Tournament time
        tournament_pool = population + offspring

        for individual in tournament_pool:
            opponents = random.sample(exclude_from_set(individual, tournament_pool), q(dimensions))
            individual.wins = reduce(operator.add,
                                     map(lambda _: 1,
                                         filter(lambda x: x.fitness > individual.fitness,
                                                opponents)),
                                     0)

        population = sorted(tournament_pool, key=lambda x: x.wins, reverse=True)[0:mu(dimensions)]
        population = sorted(population)

    return population[0].fitness


def differential_evolution(dimensions, fitness):
    population = [DEIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu(dimensions))]
    for _ in range(gens):
        next_generation = []
        for individual in population:
            child = None
            while child is None or np.isinf(child.fitness):
                mates = random.sample(exclude_from_set(individual, population), 2)
                child = individual.reproduce(mates[0], mates[1])

            if child.fitness <= individual.fitness:
                next_generation.append(child)
            else:
                next_generation.append(individual)
        population = next_generation

    best = sorted(population)[0]
    return best.fitness


def repeat(n, dimensions, fitnesses):
    for fitness in fitnesses:
        print("## " + fitness.__name__.title())
        print("### Fast EP")
        stats_ep = pd.DataFrame(columns=["Gen", "Best"])
        for i in range(n):
            best = fastEP(dimensions, fitness)
            stats_ep.loc[i] = [i, best]
            print(best) if verbose else progress_bar(i, n)
        print(stats_ep["Best"].describe())

        print("### Differential evolution")
        stats_de = pd.DataFrame(columns=["Gen", "Best"])
        for i in range(n):
            best = differential_evolution(dimensions, fitness)
            stats_de.loc[i] = [i, best]
            print(best) if verbose else progress_bar(i, n)
        print(stats_de["Best"].describe())


def progress_bar(i, n):
    proportion = (i + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(50*proportion), 100*proportion))
    sys.stdout.flush()


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
