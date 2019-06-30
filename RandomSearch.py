from multiprocessing import Pool
from GeneralNN import GeneralNeuralNetwork
from update_rule import UpdateRule
from helper.activations import sigmoid_activation
from helper.losses import MeanSquaredCost
from data.fashion_MNIST import load_data, prepare_data_for_tooc
import numpy as np
import random
import os
import warnings
from timeit import default_timer as timer

np.seterr(all='warn')
warnings.filterwarnings('error')
num_cores = 2

network_size = (784, 20, 10, 10)
mse = MeanSquaredCost()

batch_size = 3000
epochs = 5
threshold = 0.15

fitness_function = lambda x, y: x-y

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)


def fitness_evaluation(solution):
    NN = GeneralNeuralNetwork(network_size, sigmoid_activation, fitness_function, 
                                solution)

    accuracy_history = []

    for e in range(epochs):
        for b in range(x_train.shape[0] // batch_size):           
            for s in range(batch_size):
                try:
                    NN.train_network(x_train[b * batch_size + s], y_train[b * batch_size + s])
                except Warning:
                    if NN.has_invalid_values and NN.get_accuracy() < 0.15: 
                        print('Invalid values encountered')                       
                        return 0

            accuracy_history += [NN.get_accuracy()]
            if accuracy_history[-1] < 0.15:                
                break
            NN.reset_accuracy()

    print(accuracy_history[-1], ":", solution)

    return accuracy_history[-1]


if __name__ == '__main__':
    solutions = list(UpdateRule.generate_all_rules())

    print('Evaluate', str(len(solutions)), 'rules')

    # Run fitness evaluation on multiple cores
    pool = Pool(processes=num_cores)

    start = timer()

    results = pool.map(fitness_evaluation, solutions)

    end = timer()

    print(end - start)

    print(max(results))
