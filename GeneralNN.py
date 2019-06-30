import numpy as np  

"""
Implements a vanilla feed-forward neural network without bias. It uses the 
forward-reaction-backwards-feedback model and can be trained with an arbitrary 
update rule for the weights and the neuron-performances.
"""
class GeneralNeuralNetwork():
    def __init__(self, size, activation_function, feedback_function, update_rule):
        self.activation_function = activation_function
        self.feedback_function = feedback_function

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
        # feed-forward pass  
        self.get_output(input_values)

        # backward feedback and weight update
        output_performances = self.feedback_function(self.neurons[-1], correct_output_values)
        self.backpropagate_neuron_performances(output_performances) 
        self.adapt_weights()

        self.samples_trained += 1
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
        # aliases for brevity
        rule = self.update_rule
        lr = rule.performance_lr
        
        self.performances[-1] = output_performances

        for l in range(self.num_layers - 2, -1, -1):
            # the weighted sum of the performances of the neurons in the next layer
            p_out = np.dot(self.weights[l], self.performances[l+1])
            
            self.performances[l] *= 0

            if rule.is_set('y'):
                self.performances[l] += lr * rule.get('y') * self.neurons[l]
            if rule.is_set('p_out'):
                self.performances[l] += lr * rule.get('p_out') * p_out
            if rule.is_set('y^2'):
                self.performances[l] += lr * rule.get('y^2') * self.neurons[l]**2
            if rule.is_set('p_out^2'):
                self.performances[l] += lr * rule.get('p_out^2') * p_out**2
            if rule.is_set('y*p_out'):
                self.performances[l] += lr * rule.get('y*p_out') * self.neurons[l] * p_out
            if rule.is_set('y^2*p_out'):
                self.performances[l] += lr * rule.get('y^2*p_out') * self.neurons[l]**2 * p_out
            if rule.is_set('y*p_out^2'):
                self.performances[l] += lr * rule.get('y*p_out^2') * self.neurons[l] * p_out**2    

    """
    Adapts the weights according to the update rule.
    """
    def adapt_weights(self):
        # aliases for brevity
        rule = self.update_rule
        lr = rule.weight_lr

        for l in range(self.num_layers - 1):
            if rule.is_set('p1'):
                self.weights[l] += lr * rule.get('p1') * self.performances[l][np.newaxis].T
            if rule.is_set('p2'):
                self.weights[l] += lr * rule.get('p2') * self.performances[l+1]
            if rule.is_set('y1'):
                self.weights[l] += lr * rule.get('y1') * self.neurons[l][np.newaxis].T
            if rule.is_set('y2'):
                self.weights[l] += lr * rule.get('y2') * self.neurons[l+1]
            if rule.is_set('p1*y1'):
                self.weights[l] += lr * rule.get('p1*y1') * (self.performances[l] * self.neurons[l])[np.newaxis].T
            if rule.is_set('p2*y2'):
                self.weights[l] += lr * rule.get('p2*y2') * self.performances[l+1] * self.neurons[l+1]
            if rule.is_set('p1*p2'):
                self.weights[l] += lr * rule.get('p1*p2') * np.dot(self.performances[l][np.newaxis].T, self.performances[l+1][np.newaxis])
            if rule.is_set('p1*y2'):
                self.weights[l] += lr * rule.get('p1*y2') * np.dot(self.performances[l][np.newaxis].T, self.neurons[l+1][np.newaxis])
            if rule.is_set('p2*y1'):
                self.weights[l] += lr * rule.get('p2*y1') * np.dot(self.neurons[l][np.newaxis].T, self.performances[l+1][np.newaxis])
            if rule.is_set('y1*y2'):
                self.weights[l] += lr * rule.get('y1*y2') * np.dot(self.neurons[l][np.newaxis].T, self.neurons[l+1][np.newaxis])

    def reset_accuracy(self):
        self.samples_trained = 0
        self.samples_correctly_classified = 0

    def get_accuracy(self):
        return self.samples_correctly_classified / max(1, self.samples_trained)

    def has_invalid_values(self):
        return not np.isfinite(self.neurons[-1]).all()