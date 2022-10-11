import time

import numpy as np
from sklearn.svm import LinearSVC

from AIML426Project2.IDGP.IDGP_main import train, build_toolbox

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
        pop, log, hof = train(seed, build_toolbox(x_train, y_train, LinearSVC))
        endTime = time.process_time()
        trainTime = endTime - beginTime
