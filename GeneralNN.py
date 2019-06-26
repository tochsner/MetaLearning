import numpy as np  
import math         
import io         
import scipy as sp     
from scipy import stats                                                                                           


class BaseNetwork:
    def __init__(self):
        return None    
    def get_output(self, input_values):
        return None    
    def get_derivatives(self, output_derivatives):
        return None
    def train_network(self, input_values, correct_output_values):
        return None
    def apply_changes(self, learning_rate, rsegularization):
        return None


"""
Implements a vanilla feed-forward neural network. It uses the forward-reaction-backwards-feedback model
and can be trained with an arbitrary update rule for the weights and the neuron-performances.
"""
class GeneralNeuronalNetwork(BaseNetwork):
    def __init__(self, size, activation_function, feedback_function, performance_parameter, weight_parameter,
                    performance_lr, weight_lr):
        BaseNetwork.__init__(self)

        if len(size) < 2 or min(size) == 0:
            raise ValueError("Size of network is not valid.")            
        
        self.activation_function = activation_function
        self.feedback_function = feedback_function

        # Initialize network        
        self.size = size
        self.layer_count = len(size)        

        self.neurons = [np.zeros((x)) for x in size]
        self.weights = [np.random.randn(size[x], size[x + 1]) for x in range(0, self.layer_count - 1)]
        self.performances = [np.zeros((x)) for x in size]

        self.new_weights = [np.zeros((size[x], size[x + 1])) for x in range(0, self.layer_count - 1)]
      
        self.batch_size = 0

        self.performance_parameter = performance_parameter
        self.weight_parameter = weight_parameter
        self.performance_lr = performance_lr
        self.weight_lr = weight_lr
        
        self.samples_trained = 0
        self.samples_correctly_classified = 0

    def load(self, path):
        # Load weights and bias from the files in 'path'
        BaseNetwork.__init__(self)

        self.size = np.load(path + "/size.npy")
        self.weights = np.load(path + "/weights.npy")
       
        print(self.size)

    def save(self, path):
        np.save(path + "/weights", self.weights)
        np.save(path + "/size", self.size)

    def train_network(self, input_values, correct_output_values):      
        # feed-forward pass  
        self.get_output(input_values)

        self.backpropagate_neuron_performances(self.feedback_function(self.neurons[-1], correct_output_values))                
        self.adapt_weights()

        self.samples_trained += 1
        if np.argmax(self.neurons[-1]) == np.argmax(correct_output_values):
            self.samples_correctly_classified += 1

    """
    Calculates the output for a specific input.
    """
    def get_output(self, input_values):
        self.neurons[0] = input_values

        for i in range(self.layer_count - 1):
            self.neurons[i + 1] = self.activation_function(self.weights[i].T.dot(self.neurons[i]))

        return self.neurons[-1]

    """
    Backpropagates through the network to calculate the neuron-performance parameters.
    """
    def backpropagate_neuron_performances(self, output_performances):
        self.performances[-1] = output_performances

        for l in range(self.layer_count - 2, -1, -1):
            # the weighted sum of the performances of the neurons in the next layer
            p_out = np.dot(self.weights[l], self.performances[l+1])

            self.performances[l] *= 0

            if self.performance_parameter[0] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[0] * self.neurons[l]
            if self.performance_parameter[1] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[1] * p_out
            if self.performance_parameter[2] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[2] * np.square(self.neurons[l])
            if self.performance_parameter[3] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[3] * np.square(p_out)
            if self.performance_parameter[4] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[4] * self.neurons[l] * p_out
            if self.performance_parameter[5] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[5] * np.square(self.neurons[l]) * p_out
            if self.performance_parameter[6] != 0:
                self.performances[l] += self.performance_lr * self.performance_parameter[6] * self.neurons[l] * np.square(p_out)
            

    """
    Adapts the weights according to the update-rule.
    """
    def adapt_weights(self):
        for l in range(self.layer_count - 1):
            if self.weight_parameter[0] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[0] * self.performances[l][np.newaxis].T
            if self.weight_parameter[1] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[1] * self.performances[l+1]
            if self.weight_parameter[2] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[2] * self.neurons[l][np.newaxis].T
            if self.weight_parameter[3] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[3] * self.neurons[l+1]
            if self.weight_parameter[4] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[4] * (self.performances[l] * self.neurons[l])[np.newaxis].T
            if self.weight_parameter[5] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[5] * (self.performances[l+1] * self.neurons[l+1])
            if self.weight_parameter[6] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[6] * np.dot(self.performances[l][np.newaxis].T, self.performances[l+1][np.newaxis])
            if self.weight_parameter[7] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[7] * np.dot(self.performances[l][np.newaxis].T, self.neurons[l+1][np.newaxis])
            if self.weight_parameter[8] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[8] * np.dot(self.neurons[l][np.newaxis].T, self.performances[l+1][np.newaxis])
            if self.weight_parameter[9] != 0:
                self.weights[l] += self.weight_lr * self.weight_parameter[9] * np.dot(self.neurons[l][np.newaxis].T, self.neurons[l+1][np.newaxis])

    def reset_accuracy(self):
        self.samples_trained = 0
        self.samples_correctly_classified = 0

    def get_accuracy(self):
        return self.samples_correctly_classified / max(1, self.samples_trained)

    def has_invalid_values(self):
        return not np.isfinite(self.neurons[-1]).all()