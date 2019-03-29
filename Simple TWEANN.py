from GA import VanillaGA, GAParameter
from selectors import TournamentSelector
from NN import SimpleNeuronalNetwork
from helper.activations import sigmoid_activation, sigmoid_derivation
from helper.losses import MeanSquaredCost
from data.fashion_MNIST import load_data, prepare_data_for_tooc
import scipy as sp
import numpy as np
import random

iterations = 1000

population_size = 100
solution_mutation_rate = 1
weight_mutation_rate = 0.3
weight_mutation_factor = 0.5

parameters = GAParameter(population_size, solution_mutation_rate)
selector = TournamentSelector()

network_size = (784, 10, 10)
mse = MeanSquaredCost()

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

class Solu:
    def __init__(self):
        self.value = random.random()

def solution_generator():
    return SimpleNeuronalNetwork(network_size, sigmoid_activation, sigmoid_derivation, mse)

def fitness_evaluation(NN):    
    accuracy = 0        

    for s in range(x_test.shape[0]):
        output = NN.get_output(x_test[s, :])
        if np.argmax(output) == np.argmax(y_test[s, : ]):
            accuracy += 1

    return accuracy / x_test.shape[0]

def randValue(n):
    return 2 * (np.random.random(n) - 0.5)

def mutation(NN):    
    NN.weights = [w + weight_mutation_factor * sp.sparse.random(w.shape[0], w.shape[1], density=weight_mutation_rate, data_rvs=randValue).toarray() for w in NN.weights]

GA = VanillaGA(parameters, solution_generator, fitness_evaluation, mutation, selector)

GA.initialize()

for i in range(iterations):    
    GA.perform_iteration()
    print("Best accuracy after", i, "iterations:", GA.get_best_fitness(), "( Mean accuracy:", GA.get_average_fitness(), ")", flush=True)