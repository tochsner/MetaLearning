"""
Trains a simple neural network on MNIST classification, using my implementation.
"""

from data.MNIST import *
from GeneralNN import *
from helper.activations import *
from helper.losses import *
import numpy as np
from timeit import default_timer as timer

np.seterr(all='warn')

def train_model():
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)
 
    batch_size = 20
    epochs = 100
    lr = 1
    r = 0.00001

    mse = MeanSquaredCost()

    classifier = GeneralNeuronalNetwork((784, 50, 50, 10), sigmoid_activation, mse.get_derivatives, np.zeros((7)), np.zeros((10)),
                                     performance_lr=0.1, weight_lr=0.001)
    
    classifier.performance_parameter[3] = -1
    classifier.weight_parameter[8] = -1
    classifier.weight_parameter[9] = -1

    for e in range(epochs):
        for s in range(x_train.shape[0]):
            classifier.train_network(x_train[s], y_train[s])          
        
        accuracy = 0        

        for s in range(x_test.shape[0]):
            output = classifier.get_output(x_test[s, :])
            if np.argmax(output) == np.argmax(y_test[s, : ]):
                accuracy += 1                    
        
        print(accuracy / x_test.shape[0], classifier.get_accuracy(), flush = True)

        classifier.reset_accuracy()


train_model()
