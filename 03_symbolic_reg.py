"""
Build a GP system to automatically evolve a number of genetic programs for the following regression problem:
f(x) = { 1/x+sinx, x>0; 2x+x^2+3.0, x≤0 }

Inspired from https://github.com/DEAP/deap/blob/12ed30e0b8fc95c155f15dfce468b79e72153d74/examples/gp/symbreg.py
"""
import math
import operator
import random
from pprint import pprint

import numpy
import numpy as np
import pygraphviz as pgv
from deap import creator, base, tools, gp
from deap.algorithms import varOr

verbose = False
mu = 1000
p_cross = 0.9
p_mutate = 0.1
n_elite = int(mu*0.1)
epochs = 100
init_min_depth = 1
init_max_depth = 3
max_depth = 6
mutate_min_depth = 1
mutate_max_depth = 3


def create_primitive_set():
    primitives = gp.PrimitiveSetTyped("main", [float], float, "x")
    # Numeric operators
    primitives.addPrimitive(operator.add, [float, float], float)
    primitives.addPrimitive(operator.sub, [float, float], float)
    primitives.addPrimitive(operator.mul, [float, float], float)
    # Add extra operators necessary to the problem
    primitives.addPrimitive(math.sin, [float], float)

    def protectedDiv(x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 1

    primitives.addPrimitive(protectedDiv, [float, float], float)

    def square(x):
        try:
            return x ** 2
        except OverflowError:
            return float("inf")

    # Adding a power operator gave it too much power it couldn't handle (negative operands => complex)
    primitives.addPrimitive(square, [float], float)
    primitives.addEphemeralConstant("rand1", lambda: random.random() * 100, float)

    return primitives


def evaluate(individual, points):
    f = toolbox.compile(expr=individual)

    def trueF(x):
        if x > 0:
            return 1 / x + math.sin(x)
        return 2 * x + x ** 2 + 3.0

    sse = math.fsum([(f(x) - trueF(x)) ** 2 for x in points])
    # Return as a tuple for DEAP
    return sse / len(points),


def create_toolbox():
    toolbox = base.Toolbox()
    # Generate individual using half (full) and half (grow) method
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitives, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # Generate population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitives)

    # Evaluate over x = {-10,-9,-8,...,8,9,10}
    toolbox.register("evaluate", evaluate, points=[x / 5. for x in range(-50, 75)])
    toolbox.register("evaluate1", evaluate, points=[x / 5. for x in range(1, 75)])
    toolbox.register("evaluate2", evaluate, points=[x / 5. for x in range(-50, 1)])
    # Tournament selection, 3 participants
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    # Mutate new subtrees
    toolbox.register("expr_mut", gp.genFull, min_=mutate_min_depth, max_=mutate_max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    return toolbox


# Start with primitive set to feed into toolbox
primitives = create_primitive_set()
# Minimise the fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
# Set up toolbox to generate population
toolbox = create_toolbox()


def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    pops = [toolbox.population(n=mu), toolbox.population(n=mu)]
    hofs = [tools.HallOfFame(1), tools.HallOfFame(1)]
    evals = [toolbox.evaluate1, toolbox.evaluate2]

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", numpy.min)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)

    # Copied from DEAP source code (eaMuPlusLambda) with modifications indicated
    log = tools.Logbook()
    log.header = ['gen', 'nevals'] + mstats.fields

    for i, (pop, hof, evaluate) in enumerate(zip(pops, hofs, evals)):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(pop)

    # Begin the generational process
    for gen in range(0, epochs):
        for i, (pop, hof, evaluate) in enumerate(zip(pops, hofs, evals)):
            # Vary the population
            offspring = varOr(pop, toolbox, mu, p_cross, p_mutate)

            # Evaluate the individuals with an invalidated fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            hof.update(offspring)

            # MODIFIED: Select the next generation population by adding the most elite individuals from the last population,
            # then populate with offspring
            pop[:] = tools.selBest(pop, n_elite) + tools.selBest(offspring, mu - n_elite)

            # Update the statistics with the new pop
            record = mstats.compile(pop)
            log.record(gen=gen, pop=i, nevals=len(invalid_ind), **record)
            if verbose:
                print(log.stream)

    g = pgv.AGraph()
    g.node_attr['style'] = 'filled'
    g.add_node(0)
    g.get_node(0).attr["label"] = "if"
    g.get_node(0).attr["fillcolor"] = "#cccccc"
    g.add_node(1)
    g.get_node(1).attr["label"] = "x>0"
    g.get_node(1).attr["fillcolor"] = "#cccccc"
    g.add_edge(0, 1)

    colors = ["#f597bb", "#a5abee"]
    for pop, hof in enumerate(hofs):
        best_program = toolbox.compile(expr=hof.items[0])
        print("## Best individual in population", str(pop + 1))
        print("Fitness =", hof.keys[0])
        print("Size =", len(hof.items[0]))
        if verbose:
            print(str(hof.items[0]))
            pprint([best_program(x / 5.) for x in range(-25, 75)])

        nodes, edges, labels = gp.graph(hof.items[0])
        base = g.number_of_nodes()
        g.add_nodes_from([base + i for i in nodes])
        g.add_edges_from([(base + i, base + j) for (i, j) in edges])
        for i in labels:
            n = g.get_node(base + i)
            if isinstance(labels[i], float):
                labels[i] = round(labels[i], 2)
            n.attr["label"] = labels[i]
            n.attr["fillcolor"] = colors[pop]
        g.add_edge(0, base)

    g.layout(prog="dot")
    g.draw("out/part3/"+str(seed)+"_tree.png")


if __name__ == '__main__':
    for seed in range(5):
        print("# Seed", str(seed))
        main(seed)
