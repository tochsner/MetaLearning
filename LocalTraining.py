from NN import SimpleNeuronalNetwork

"""
Adapts the weights of a SimpleNeuralNetwork in the following fashion:
w = w + func(w, y_1, y_2)
"""
def adapt_weights(func, NN):
    for l in range(NN.layer_count - 1):
        for n in range(NN.size[l]):
            NN.weights[l][n] += func(NN.weights[l][n], 
                                        NN.neurons[l][n],
                                        NN.neurons[l+1])