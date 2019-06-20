from GA import VanillaGA, GAParameter
from selectors import TournamentSelector
from NN import SimpleNeuronalNetwork
from helper.activations import sigmoid_activation, sigmoid_derivation
from helper.losses import MeanSquaredCost
from LocalTraining import adapt_weights
from data.fashion_MNIST import load_data, prepare_data_for_tooc
import numpy as np
import random

iterations = 1000

population_size = 20
solution_mutation_rate = 1

parameters = GAParameter(population_size, solution_mutation_rate)
selector = TournamentSelector()

network_size = (784, 10, 10)
mse = MeanSquaredCost()

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

class Solution:
    def __init__(self):
        self.parameter = [random.random()] + [random.choice([0,1]) for x in range(6)]

"""
Returns the function for adapting the weights.
The parameters are the following coefficients:
f = lr * (a0*w + a1*x1 + a2*x2 + a3*w*x1 + a4*w*x2 + a5*x1*x2)
"""
def get_function(parameter):
    lr, a0, a1, a2, a3, a4, a5 = parameter
    return lambda w, x1, x2: lr * (a0*w + a1*x1 + a2*x2 + a3*w*x1 + a4*w*x2 + a5*x1*x2)

def solution_generator():
    return Solution()

def mutation(solution):    
    mutation_index = random.choice(range(1, 7))    
    solution.parameter[mutation_index] = 1 - solution.parameter[mutation_index]

    return solution

def fitness_evaluation(solution):
    func = get_function(solution.parameter)        
    NN = SimpleNeuronalNetwork(network_size, sigmoid_activation, sigmoid_derivation, mse)

    for s in range(500):
        NN.get_output(x_train[s, :])
        adapt_weights(func, NN)
          
    accuracy = 0

    for s in range(x_test.shape[0]):
        output = NN.get_output(x_test[s, :])
        if np.argmax(output) == np.argmax(y_test[s, : ]):
            accuracy += 1

    return accuracy / x_test.shape[0]

GA = VanillaGA(parameters, solution_generator, fitness_evaluation, mutation, selector)

GA.initialize()

for i in range(iterations):    
    GA.perform_iteration()
    print("Best accuracy after", i, "iterations:", GA.get_best_fitness(), "( Mean accuracy:", GA.get_average_fitness(), ")", flush=True)