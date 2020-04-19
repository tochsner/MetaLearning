import functools

import numpy as np

import searchHelper.search_parameter as SP
import updateModel.model_parameter as MP
from data.fashion_MNIST import load_data, prepare_data_for_tooc
from helper.general_NN import GeneralNeuralNetwork
from helper.history import HistoryItem

# enable that numpy throws an exception for floating point errors
np.seterr(all='raise')
np.seterr(under='ignore')

data = load_data()
(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)


def get_evaluator(queue):
    return functools.partial(evaluate, queue=queue)


"""
Evaluates an update rule on the FashionMNIST dataset. Early stopping according
to the parameters in parameter.py is used.

This method is designed for concurrent use, where the results are appended to
a shared queue.
"""
def evaluate(rule, queue):
    NN = GeneralNeuralNetwork(SP.NETWORK_SIZE, MP.ACTIVATION_FUNCTION,
                              MP.OUTPUT_PERFORMANCE_FUNCTION, rule)

    accuracy_history = []

    for e in range(SP.MAX_EPOCHS):
        for b in range(x_train.shape[0] // SP.BATCH_SIZE):
            for s in range(SP.BATCH_SIZE):
                try:
                    NN.train_network(x_train[b * SP.BATCH_SIZE + s], y_train[b * SP.BATCH_SIZE + s])
                except FloatingPointError:
                    if NN.has_invalid_values():
                        history_item = HistoryItem(rule, accuracy_history=[0], produced_invalid_values=True)
                        queue.put(history_item)
                        return 0

            accuracy_history.append(NN.get_accuracy())

            # early stopping
            if accuracy_history[-1] < SP.ACCURACY_THRESHOLD:
                history_item = HistoryItem(rule, accuracy_history)
                queue.put(history_item)
                return max(accuracy_history)

            NN.reset_accuracy()

    history_item = HistoryItem(rule, accuracy_history)
    queue.put(history_item)

    return max(accuracy_history)
