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
Implements a vanilla feed-forward neural network, trained with mini-batch stochastic-gradient-descent.
Uses l2-regularization to minimize overfitting. 
"""
class SimpleNeuronalNetwork(BaseNetwork):
    def __init__(self, size, activation_function, activation_derivative, cost_function):
        BaseNetwork.__init__(self)

        if len(size) < 2 or min(size) == 0:
            raise ValueError("Size of network is not valid.")            

        self.size = size
        
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        self.cost_function = cost_function

        # Initialize network        
        self.layer_count = len(size)

        self.neurons = [np.zeros((x)) for x in size]
        self.bias = [np.zeros((x)) for x in size]
        self.weights = [np.random.randn(size[x], size[x + 1]) for x in range(0, self.layer_count - 1)]

        self.derivatives = [np.zeros((x)) for x in size]

        self.new_weights = [np.zeros((size[x], size[x + 1])) for x in range(0, self.layer_count - 1)]
        self.new_bias = [np.zeros((x)) for x in size]
        self.batch_size = 0

    def load(self, path):
        # Load weights and bias from the files in 'path'
        BaseNetwork.__init__(self)

        self.size = np.load(path + "/size.npy")
        self.weights = np.load(path + "/weights.npy")
        self.bias = np.load(path + "/bias.npy")

        print(self.size)

    def save(self, path):
        np.save(path + "/weights", self.weights)
        np.save(path + "/bias", self.bias)
        np.save(path + "/size", self.size)

    def train_network(self, input_values, correct_output_values):      
        # feed-forward pass  
        self.get_output(input_values)

        # backpropagate and return gradient of the cost in respect to the input values
        return self.get_derivatives(self.cost_function.get_derivatives(self.neurons[-1], correct_output_values))

    """
    Calculates the output for a specific input.
    """
    def get_output(self, input_values):
        self.neurons[0] = input_values

        for i in range(self.layer_count - 1):
            self.neurons[i + 1] = self.activation_function(self.weights[i].T.dot(self.neurons[i]) + self.bias[i + 1])

        return self.neurons[-1]

    """
    Backpropagates through the network, but doesn't yet change the weights.
    """
    def get_derivatives(self, output_derivatives):
        self.derivatives[-1] = self.activation_derivative(self.neurons[-1]) * output_derivatives
                
        self.new_bias[-1] += self.derivatives[-1]
       
        for i in range(self.layer_count - 2, 0, -1):        
            self.new_weights[i] += (self.neurons[i][np.newaxis].T * self.derivatives[i + 1])            
            self.derivatives[i] = self.weights[i].dot(self.derivatives[i + 1]) * self.activation_derivative(self.neurons[i])            
            self.new_bias[i] += self.derivatives[i]

        self.new_weights[0] += (self.neurons[0][np.newaxis].T * self.derivatives[1])
        self.derivatives[0] = self.weights[0].dot(self.derivatives[1])

        self.batch_size += 1

        return self.derivatives[0]


    """
    Changes the weights and biases after a batch
    """
    def apply_changes(self, learning_rate, regularization):
        self.weights = [(w - nw * (learning_rate / self.batch_size) - regularization * w) for w, nw in zip(self.weights, self.new_weights)]
        self.bias = [(b - nb * (learning_rate / self.batch_size)) for b, nb in zip(self.bias, self.new_bias)]

        self.new_weights = [np.zeros(x.shape) for x in self.new_weights]
        self.new_bias = [np.zeros(x.shape) for x in self.new_bias]

        self.batch_size = 0

    def evaluate(self, test_data):               
        test_results = [(np.argmax(self.get_output(x[:, 0])), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        