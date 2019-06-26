from GeneralNN import GeneralNeuronalNetwork
from helper.activations import sigmoid_activation
from helper.losses import MeanSquaredCost
from data.fashion_MNIST import load_data, prepare_data_for_tooc
import numpy as np
import random
import os
from multiprocessing import Pool
import warnings
from timeit import default_timer as timer
import csv

np.seterr(all='warn')
warnings.filterwarnings('error')
num_cores = 2

network_size = (784, 50, 10)
mse = MeanSquaredCost()

batch_size = 5000
epochs = 10

elite_threshold = 0.7

weight_variables = 10
performance_variables = 7

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)


def get_solution(path):
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=';')
        for line in reader:
            if float(line[-1]) > 0.7:
                performance_lr = float(line[0])
                performance_param = list(map(int, line[1:8]))
                weight_lr = float(line[8])
                weight_param = list(map(int, line[9:19]))

                yield performance_param, weight_param, performance_lr, weight_lr

                for i in range(10):
                    performance_lr = random.random() ** 2
                    weight_lr = random.random() ** 2

                    yield performance_param, weight_param, performance_lr, weight_lr


def fitness_evaluation(solution):
    performance_param, weight_param, performance_lr, weight_lr = solution
    NN = GeneralNeuronalNetwork(network_size, sigmoid_activation, mse.get_derivatives, 
                                performance_param, weight_param, performance_lr, weight_lr)

    accuracy_history = []

    for e in range(epochs):
        for b in range(x_train.shape[0] // batch_size):           
            for s in range(batch_size):
                try:
                    NN.train_network(x_train[b * batch_size + s], y_train[b * batch_size + s])
                except Warning:
                    if NN.has_invalid_values and NN.get_accuracy() < 0.15:                        
                        return 0

            accuracy_history += [NN.get_accuracy()]
            if accuracy_history[-1] < 0.15:
                break
            NN.reset_accuracy()

    with open("logs/elite.csv", "a+") as file:
        file.write(";".join(map(str,[performance_lr] + performance_param + [weight_lr] + weight_param + accuracy_history)) + "\n")

    print(accuracy_history[-1])

    return accuracy_history[-1]


if __name__ == '__main__':
    paths = ['logs/10564.csv', 'logs/11020.csv', 'logs/13948.csv', 'logs/15152.csv', 'logs/15472.csv', 
                'logs/15768.csv']

    solutions = []

    for path in paths:
        solutions += list(get_solution(path))

    print('Evaulate ', len(solutions), 'solutions')

    # Run fitness evaluation on multiple cores
    pool = Pool(processes=num_cores)

    results = pool.map(fitness_evaluation, solutions)