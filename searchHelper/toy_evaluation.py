import numpy as np

import updateModel.model_parameter as MCN
from helper.general_NN import GeneralNeuralNetwork


np.seterr(all='raise')
np.seterr(under='ignore')

output_size = (20,)
epochs = 25
trials = 20


"""
Evaluates an update rule on a toy problem: the network should learn to always 
output a certain, randomly generated output pattern.

This method is designed for concurrent use, where the results are appended to
a shared queue.
"""
def evaluate(rule):
    NN = GeneralNeuralNetwork((1,) + output_size, MCN.ACTIVATION_FUNCTION,
                              MCN.OUTPUT_PERFORMANCE_FUNCTION, rule)

    accuracy = 0

    for i in range(trials):
        correct_output = np.random.randint(0, 2, output_size)

        for e in range(epochs):  
            try:                         
                NN.train_network(np.ones(1), correct_output)
            except:
                pass

        output = NN.get_output(np.ones(1))

        for o, c_o in zip(list(output), list(correct_output)):
            if o > 0.5 and c_o > 0.5:
                accuracy += 1
            if o < 0.5 and c_o < 0.5:
                accuracy += 1
    
    return accuracy / output_size[0] / trials
