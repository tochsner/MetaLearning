from multiprocessing import Pool
from GeneralNN import GeneralNeuronalNetwork
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
num_cores = 4

network_size = (784, 50, 20, 10)
mse = MeanSquaredCost()

batch_size = 3000
epochs = 5
threshold = 0.15

weight_variables = 10
performance_variables = 7

max_chosen_weight_variables = 4
max_chosen_performance_variables = 3

fitness_function = lambda x, y: x-y

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)


def generate_solution():
    performance_param = [0 for x in range(performance_variables)]
    weight_param = [0 for x in range(weight_variables)]

    for i in range(random.randint(1, max_chosen_performance_variables)):
        performance_param[random.randint(0, performance_variables-1)] = random.choice([1,-1])

    for i in range(random.randint(1, max_chosen_weight_variables)):
        weight_param[random.randint(0, weight_variables-1)] = random.choice([1,-1])

    performance_lr = random.random() ** 2
    weight_lr = random.random() ** 2

    return performance_param, weight_param, performance_lr, weight_lr


def fitness_evaluation(solution):
    performance_param, weight_param, performance_lr, weight_lr = solution
    NN = GeneralNeuronalNetwork(network_size, sigmoid_activation, fitness_function , 
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

    with open("logs/784-50-20-10 " + str(os.getpid()) + ".csv", "a+") as file:
        file.write(";".join(map(str,[performance_lr] + performance_param + [weight_lr] + weight_param + accuracy_history)) + "\n")

    print(accuracy_history[-1])

    return accuracy_history[-1]


if __name__ == '__main__':
    solutions = [generate_solution() for x in range(100000)]

    # Run fitness evaluation on multiple cores
    pool = Pool(processes=num_cores)

    start = timer()

    results = pool.map(fitness_evaluation, solutions)

    end = timer()

    print(end - start)

    print(max(results))
