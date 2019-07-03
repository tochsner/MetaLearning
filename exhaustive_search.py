from multiprocessing.pool import Pool
from multiprocessing import SimpleQueue, Manager
import numpy as np
from os import makedirs, path
from shutil import copyfile
import warnings
from time import sleep
from timeit import default_timer as Timer

import parameter as CN
from data.fashion_MNIST import load_data, prepare_data_for_tooc
from GeneralNN import GeneralNeuralNetwork
from update_rule import UpdateRule
from history import HistoryItem, HistoryManager
from helper.activations import sigmoid_activation

np.seterr(all='raise')
np.seterr(under='ignore')

def evaluate_update_rule(parameter):
    queue, rule, x_train, y_train = parameter 

    NN = GeneralNeuralNetwork(CN.NETWORK_SIZE, sigmoid_activation, CN.OUTPUT_FITNESS_FUNCTION, rule)

    accuracy_history = []

    for e in range(CN.MAX_EPOCHS):
        for b in range(x_train.shape[0] // CN.BATCH_SIZE):
            for s in range(CN.BATCH_SIZE):
                try:
                    NN.train_network(x_train[b * CN.BATCH_SIZE + s], y_train[b * CN.BATCH_SIZE + s])
                except FloatingPointError:                                          
                    if NN.has_invalid_values(): 
                        history_item = HistoryItem(rule, [0], True)                    
                        queue.put(history_item)                        
                        return 0

            accuracy_history += [NN.get_accuracy()]

            if accuracy_history[-1] < CN.ACCURACY_THRESHOLD:                
                history_item = HistoryItem(rule, accuracy_history, False)
                queue.put(history_item)
                return max(accuracy_history)

            NN.reset_accuracy()

    history_item = HistoryItem(rule, accuracy_history, False)
    queue.put(history_item)

    return max(accuracy_history)

if __name__ == '__main__':
    DIRECTORY_PATH = 'logs/' + CN.EXPERIMENT_NAME
    LOG_PATH = DIRECTORY_PATH + '/logs.pkl'
    PARAMETER_PATH = DIRECTORY_PATH + '/parameter.py'

    if not path.exists(DIRECTORY_PATH):
        makedirs(DIRECTORY_PATH)

    copyfile('parameter.py', PARAMETER_PATH)

    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)
    
    manager = Manager()
    queue = manager.Queue()
    
    all_rules = list(UpdateRule.generate_all_rules())
    all_solutions = [(queue, rule, x_train, y_train) for rule in all_rules]

    history_manager = HistoryManager(queue, LOG_PATH, len(all_solutions))    

    print('Evaluate', len(all_rules), 'rules')

    pool = Pool(processes=CN.NUM_OF_CORES)
    results = pool.map(evaluate_update_rule, all_solutions)

    history_manager.end()

    index_list = range(len(results))    
    print('Finished. Maximum achieved accuracy ', max(results), 'with', 
            all_solutions[max(index_list, key=lambda x: results[x])][1])