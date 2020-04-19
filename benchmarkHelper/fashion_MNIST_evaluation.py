from  data.fashion_MNIST import load_data, prepare_data_for_tooc

from benchmarkHelper import benchmark_parameter as BCN
import updateModel.model_parameter as MCN
from helper.general_NN import GeneralNeuralNetwork
from helper.history import HistoryItem

data = load_data()

(x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)


def evaluate(parameter):
    queue, rule = parameter 

    NN = GeneralNeuralNetwork(BCN.NETWORK_SIZE, MCN.ACTIVATION_FUNCTION,
                              MCN.OUTPUT_PERFORMANCE_FUNCTION, rule)

    test_accuracy, training_accuracy = NN.evaluate(x_train, y_train, x_test, y_test, BCN.EPOCHS)

    history_item = HistoryItem(rule, [training_accuracy], False)
    history_item.test_accuracy = test_accuracy

    queue.put(history_item)

    return test_accuracy
