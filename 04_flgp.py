# python packages
import operator
import random
import time
import warnings

import numpy as np
from deap import base, creator, tools, gp
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import IDGP.evalGP_main as evalGP
import IDGP.feature_function as fe_fs
from IDGP import gp_restrict
from IDGP.strongGPDataType import Int1, Int2, Int3, Img, Region, Vector

warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')

# parameters:
population = 100
generation = 50
cxProb = 0.8
mutProb = 0.2
elitismProb = 0.05
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8


def build_primitives(data, bound1, bound2):
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
    # Feature concatenation
    pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
    pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')
    # Global feature extraction
    pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
    pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
    # Local feature extraction
    pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
    pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
    # Region detection operators
    pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
    # Terminals
    pset.renameArguments(ARG0='Grey')
    pset.addEphemeralConstant('X_'+data, lambda: random.randint(0, bound1 - 20), Int1)
    pset.addEphemeralConstant('Y_'+data, lambda: random.randint(0, bound2 - 20), Int2)
    pset.addEphemeralConstant('Size_'+data, lambda: random.randint(20, 51), Int3)

    return pset


# fitness evaluation
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def evaluate(individual, x, y, classifier, cv=True, **kwargs):
    # print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y)):
        train_tf.append(np.asarray(func(x[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # print(train_norm.shape)
    clf = classifier(**kwargs)
    if cv:
        metric = round(cross_val_score(clf, train_norm, y, scoring="f1", cv=3).mean(), 6),
    else:
        clf.fit(train_norm, y)
        pred = clf.predict(train_norm)
        metric = (f1_score(pred, y), clf.score(train_norm, y))
    return metric


def build_toolbox(data, x_train, y_train, classifier, **kwargs):
    bound1, bound2 = x_train[0, :, :].shape
    pset = build_primitives(data, bound1, bound2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mapp", map)

    # genetic operator
    toolbox.register("evaluate", evaluate, x=x_train, y=y_train, classifier=classifier, cv=True, **kwargs)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

    return toolbox


def train(randomSeeds, toolbox):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof


def transform(individual, toolbox, x, y):
    func = toolbox.compile(expr=individual)

    data = []
    for i in range(0, len(y)):
        data.append(np.asarray(func(x[i, :, :])))
    data = np.asarray(data)
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(np.asarray(data))

    return np.append(norm_data, y.reshape((-1, 1)), axis=1)


if __name__ == "__main__":
    path = "data/FEI-dataset/"

    datasets = ['f1', 'f2']
    for data in datasets:
        x_train = np.load(path + data + '/' + data + '_train_data.npy') / 255.0
        y_train = np.load(path + data + '/' + data + '_train_label.npy')
        x_test = np.load(path + data + '/' + data + '_test_data.npy') / 255.0
        y_test = np.load(path + data + '/' + data + '_test_label.npy')

        print(x_train.shape)

        seed = 0
        beginTime = time.process_time()
        toolbox = build_toolbox(data, x_train, y_train, SVC)
        pop, log, hof = train(seed, toolbox)
        endTime = time.process_time()
        trainTime = endTime - beginTime
        print(data + " - training time (s): " + str(trainTime))

        best = hof.items[0]
        features_train = transform(best, toolbox, x_train, y_train)
        np.savetxt("out/part4/{}_features_train.csv".format(data), features_train, delimiter=",")
        features_test = transform(best, toolbox, x_test, y_test)
        np.savetxt("out/part4/{}_features_test.csv".format(data), features_test, delimiter=",")

        print(data + " test score: " + str(evaluate(best, x_test, y_test, SVC, False)))
        print("Best individual: " + str(best))

