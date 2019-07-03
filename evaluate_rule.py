from multiprocessing.pool import Pool
from multiprocessing import SimpleQueue, Manager
import numpy as np
from os import makedirs, path
from shutil import copyfile
import warnings
from time import sleep
from timeit import default_timer as Timer

import parameter as CN
from data.fashion_MNIST import load_data, prepare_data_for_tooc
from GeneralNN import GeneralNeuralNetwork
from update_rule import UpdateRule
from history import HistoryItem, HistoryManager
from helper.activations import sigmoid_activation

NETWORK_SIZE = (784, 40, 20, 10)
EPOCHS = 20

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

def evaluate_update_rule(rule):
    NN = GeneralNeuralNetwork(NETWORK_SIZE, sigmoid_activation, CN.OUTPUT_FITNESS_FUNCTION, rule)

    for e in range(EPOCHS):
        for s in range(x_train.shape[0]):            
            try:
                NN.train_network(x_train[s], y_train[s])
            except Warning:
                if NN.has_invalid_values() and NN.get_accuracy() < 0.15: 
                    print("Invalid values encountered.")
                    return

        accuracy = 0
        
        for s in range(x_test.shape[0]):
            output = NN.get_output(x_test[s])
            if np.argmax(output) == np.argmax(y_test[s]):
                accuracy += 1

        print(accuracy / x_test.shape[0])

rule = UpdateRule()
rule.set('y^2', -1)
rule.set('y*p_out^2', 1)

rule.set('p2', 1)
rule.set('p1*p2', -1)

rule.performance_lr = 0.95
rule.weight_lr = 0.37

print('Evaluate ', rule)

evaluate_update_rule(rule)
