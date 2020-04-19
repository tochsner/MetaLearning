"""
Describes the general model used to adapt the weights.
"""

import numpy as np

ACTIVATION_FUNCTION = lambda x: 1.0 / (1 + np.exp(-x))
OUTPUT_PERFORMANCE_FUNCTION = lambda y, y_target: y - y_target

# variables of the weight-update rule
PERFORMANCE_VARIABLES = ['y',
                        'p_out',
                        'y^2',
                        'p_out^2',
                        'y*p_out',
                        'y^2*p_out',
                        'y*p_out^2']
# variables of the performance-update rule
WEIGHT_VARIABLES = ['p1',
                    'p2',
                    'y1',
                    'y2',
                    'p1*y1',
                    'p2*y2',
                    'p1*p2',
                    'p1*y2',
                    'p2*y1',
                    'y1*y2']

# number of summands of the weight-update rule
NUM_WEIGHT_SUMMANDS = len(WEIGHT_VARIABLES)
# number of summands of the performance-update rule
NUM_PERFORMANCE_SUMMANDS = len(PERFORMANCE_VARIABLES)


"""
Backpropagates through a general NN to calculate the neuron-performance parameters 
according to the update rule.
"""
def backpropagate_neuron_performances(NN, output_performances):
    # aliases for brevity
    rule = NN.update_rule
    lr = rule.performance_lr
    
    NN.performances[-1] = output_performances

    for l in range(NN.num_layers - 2, -1, -1):
        # the weighted sum of the performances of the neurons in the next layer
        p_out = np.dot(NN.weights[l], NN.performances[l+1])
        y = NN.neurons[l]
        
        NN.performances[l] *= 0

        if rule.is_set('y'):
            NN.performances[l] += lr * rule['y'] * y
        if rule.is_set('p_out'):
            NN.performances[l] += lr * rule['p_out'] * p_out
        if rule.is_set('y^2'):
            NN.performances[l] += lr * rule['y^2'] * y**2
        if rule.is_set('p_out^2'):
            NN.performances[l] += lr * rule['p_out^2'] * p_out**2
        if rule.is_set('y*p_out'):
            NN.performances[l] += lr * rule['y*p_out'] * y * p_out
        if rule.is_set('y^2*p_out'):
            NN.performances[l] += lr * rule['y^2*p_out'] * y**2 * p_out
        if rule.is_set('y*p_out^2'):
            NN.performances[l] += lr * rule['y*p_out^2'] * y * p_out**2


"""
Adapts the weights of a general NN according to the update rule.
"""
def adapt_weights(NN):
    # aliases for brevity
    rule = NN.update_rule
    lr = rule.weight_lr

    for l in range(NN.num_layers - 1):
        if rule.is_set('p1'):
            NN.weights[l] += lr * rule['p1'] * NN.performances[l][np.newaxis].T
        if rule.is_set('p2'):
            NN.weights[l] += lr * rule['p2'] * NN.performances[l+1]
        if rule.is_set('y1'):
            NN.weights[l] += lr * rule['y1'] * NN.neurons[l][np.newaxis].T
        if rule.is_set('y2'):
            NN.weights[l] += lr * rule['y2'] * NN.neurons[l+1]
        if rule.is_set('p1*y1'):
            NN.weights[l] += lr * rule['p1*y1'] * (NN.performances[l] * NN.neurons[l])[np.newaxis].T
        if rule.is_set('p2*y2'):
            NN.weights[l] += lr * rule['p2*y2'] * NN.performances[l+1] * NN.neurons[l+1]
        if rule.is_set('p1*p2'):
            NN.weights[l] += lr * rule['p1*p2'] * np.dot(NN.performances[l][np.newaxis].T, NN.performances[l+1][np.newaxis])
        if rule.is_set('p1*y2'):
            NN.weights[l] += lr * rule['p1*y2'] * np.dot(NN.performances[l][np.newaxis].T, NN.neurons[l+1][np.newaxis])
        if rule.is_set('p2*y1'):
            NN.weights[l] += lr * rule['p2*y1'] * np.dot(NN.neurons[l][np.newaxis].T, NN.performances[l+1][np.newaxis])
        if rule.is_set('y1*y2'):
            NN.weights[l] += lr * rule['y1*y2'] * np.dot(NN.neurons[l][np.newaxis].T, NN.neurons[l+1][np.newaxis])
