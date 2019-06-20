import winsound
from GA import VanillaGA, GAParameter
from selectors import TournamentSelector
from NN import SimpleNeuronalNetwork
from helper.activations import sigmoid_activation, sigmoid_derivation
from helper.losses import MeanSquaredCost
from LocalTraining import adapt_weights
from data.fashion_MNIST import load_data, prepare_data_for_tooc
import numpy as np
import itertools
import random
from multiprocessing import Pool

num_cores = 4

network_size = (784, 20, 10)
mse = MeanSquaredCost()

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

"""
Returns the function for adapting the weights.
The parameters are the following coefficients:
f = lr * (a0*w + a1*x1 + a2*x2 + a3*w*x1 + a4*w*x2 + a5*x1*x2 + a6*w*x1*x2)
"""
def get_function(parameter):
    lr, a0, a1, a2, a3, a4, a5, a6 = parameter
    return lambda w, x1, x2: lr * (a0*w + a1*x1 + a2*x2 + a3*w*x1 + a4*w*x2 + a5*x1*x2 + a6*w*x1*x2)


def fitness_evaluation(solution):
    func = get_function(solution)        
    NN = SimpleNeuronalNetwork(network_size, sigmoid_activation, sigmoid_derivation, mse)
    NN2 = SimpleNeuronalNetwork((10,10), sigmoid_activation, sigmoid_derivation, mse)

    for s in range(2000):
        NN.get_output(x_train[s, :])
        adapt_weights(func, NN)

    for s in range(10000):
        embedding = NN.get_output(x_train[s])
        NN2.train_network(embedding, y_train[s])

        if (s % 10 == 0):
            NN2.apply_changes(1, 1e-5)      
          
    accuracy = 0

    for s in range(x_test.shape[0]):
        embedding = NN.get_output(x_test[s])
        output = NN2.get_output(embedding)
        if np.argmax(output) == np.argmax(y_test[s]):
            accuracy += 1

    accuracy = accuracy / x_test.shape[0]

    print(accuracy, solution)

    return accuracy


if __name__ == '__main__':  

    # Generate all solutions
    solutions = []

    solutions += [[1e-6,] + list(x) for x in itertools.product([0,1,-1], repeat=7)]
    solutions += [[1e-5,] + list(x) for x in itertools.product([0,1,-1], repeat=7)]
    solutions += [[1e-4,] + list(x) for x in itertools.product([0,1,-1], repeat=7)]
    solutions += [[1e-3,] + list(x) for x in itertools.product([0,1,-1], repeat=7)]

    print("There are ", len(solutions), "solutions to try out.")

    # Run fitness evaluation on multiple cores
    pool = Pool(processes=num_cores)
    results = pool.map(fitness_evaluation, solutions)

    with open('results.txt', 'a+') as f:
        f.write("\n**************")
        for item in zip(solutions, results):
            f.write("\n" + str(item))