import random

import numpy as np


class EPIndividual:
    epsilon = 0.1
    var_control = 1

    def __init__(self, n, evaluator, xmin=-1, xmax=1):
        self.vector = np.array(random.uniform(xmin, xmax) for _ in range(n))
        self.variance = np.array(random.uniform(xmin, xmax) for _ in range(n))

    def update(self):
        vector_increment = np.array(random.normalvariate(0, 1) for _ in range(len(self.vector)))
        vector_increment *= np.sqrt(self.variance)
        self.vector += vector_increment

        variance_increment = np.array(random.normalvariate(0, 1) for _ in range(len(self.variance)))
        variance_increment *= np.sqrt(self.variance * self.var_control)
        self.variance += variance_increment


def rosenbrock(individual):
    fitness = 0
    if any(x < -30 or x > 30 for x in individual.vector):
        return np.inf,

    for d in range(0, len(individual.vector) - 1):
        fitness += 100 * (individual.vector[d] ** 2 - individual.vector[d + 1] ** 2) ** 2 + (
                    individual.vector[d] - 1) ** 2
    return fitness,


def griewank(individual):
    summation = 0
    product = 1
    if any(x < individual.xmin or x > individual.xmax for x in individual):
        return np.inf,

    for d in range(0, len(individual)):
        summation += individual[d] ** 2
        product *= np.cos(individual[d] / np.sqrt(d + 1))

    return summation / 4000 - product + 1,


def metaEP(dimensions, fitness):
    mu = 20
    population = [EPIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu)]


def differential_evolution(dimensions, fitness):
    mu = 50
    population = [EPIndividual(dimensions, fitness, xmin=-30, xmax=30) for _ in range(mu)]


def repeat(n, dimensions, fitnesses):
    for fitness in fitnesses:
        print("## " + fitness.__name__)
        for _ in range(n):
            print("### Meta-EP")
            metaEP(dimensions, fitness)

        for _ in range(n):
            print("### Differential evolution")
            differential_evolution(dimensions, fitness)


def main(seed=None):
    repeats = 30

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Rosenbrock function at D = 20
    print("# D = 20")
    repeat(30, 20, [rosenbrock, griewank])

    print("# D = 50")
    # Griewank function at D = 20 and D = 50
    repeat(30, 50, [griewank])


if __name__ == "__main__":
    main(seed=0)
