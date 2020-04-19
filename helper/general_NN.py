import numpy as np 
import updateModel.model_parameter as MP


"""
Implements a vanilla feed-forward neural network without bias. It uses the 
forward-reaction-backwards-feedback model and can be trained with an arbitrary 
update rule for the weights and the neuron-performances.
"""
class GeneralNeuralNetwork():

    def __init__(self, size, activation_function, output_perfomance_function, update_rule):
        self.activation_function = activation_function
        self.output_perfomance_function = output_perfomance_function

        self.update_rule = update_rule
        
        self.samples_trained = 0
        self.samples_correctly_classified = 0

        # initialize network        
        self.size = size
        self.num_layers = len(size)        

        self.neurons = [np.zeros(x) for x in size]
        self.weights = [np.random.randn(size[x], size[x + 1]) for x in range(self.num_layers - 1)]
        self.performances = [np.zeros(x) for x in size]

        self.new_weights = [np.zeros((size[x], size[x + 1])) for x in range(self.num_layers - 1)]        

    def load(self, path):
        self.weights = np.load(path)
        self.size = [w.shape[0] for w in self.weights] + [self.weights[-1].shape[1]]
       
    def save(self, path):
        np.save(path, self.weights)

    def train_network(self, input_values, correct_output_values): 
        self.samples_trained += 1

        # feed-forward pass  
        self.get_output(input_values)

        # backward feedback and weight update
        output_performances = self.output_perfomance_function(self.neurons[-1], correct_output_values)
        self.backpropagate_neuron_performances(output_performances) 
        self.adapt_weights()
        
        if np.argmax(self.neurons[-1]) == np.argmax(correct_output_values):
            self.samples_correctly_classified += 1

    """
    Calculates the output for a specific input.
    """
    def get_output(self, input_values):
        self.neurons[0] = input_values

        for i in range(self.num_layers - 1):
            self.neurons[i + 1] = self.activation_function(self.weights[i].T.dot(self.neurons[i]))

        return self.neurons[-1]

    """
    Backpropagates through the network to calculate the neuron-performance parameters 
    according to the update rule.
    """
    def backpropagate_neuron_performances(self, output_performances):
        MP.backpropagate_neuron_performances(self, output_performances)

    """
    Adapts the weights according to the update rule.
    """
    def adapt_weights(self):
        MP.adapt_weights(self)

    """
    Trains the network for a given number of epochs and evaluate its accuracy on the
    training and test set.
    """
    def evaluate(self, x_train, y_train, x_test, y_test, epochs):
        # training
        for e in range(epochs):
            for s in range(x_train.shape[0]):
                self.train_network(x_train[s], y_train[s])

        # evaluate accuracy on the training set
        training_accuracy = 0

        for s in range(x_train.shape[0]):
            output = self.get_output(x_train[s])
            correct_output = y_train[s]

            if np.argmax(output) == np.argmax(correct_output):
                training_accuracy += 1   

        training_accuracy /= x_train.shape[0]

        # evaluate accuracy on the test set
        test_accuracy = 0

        for s in range(x_test.shape[0]):
            output = self.get_output(x_test[s])
            correct_output = y_test[s]

            if np.argmax(output) == np.argmax(correct_output):
                test_accuracy += 1   

        test_accuracy /= x_test.shape[0]

        return test_accuracy, training_accuracy

    def reset_accuracy(self):
        self.samples_trained = 0
        self.samples_correctly_classified = 0

    def get_accuracy(self):
        return self.samples_correctly_classified / max(1, self.samples_trained)

    def has_invalid_values(self):
        return not np.isfinite(self.neurons[-1]).all()
