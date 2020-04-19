"""
Tries all possible update rules according to the parameters specified
in parameter.py. As evaluation, the FashionMNIST dataset is used.

The search runs in parallel, with the results saved in a shared Queue
and stored periodically on a disc by the history object.
"""

from os import makedirs, path
from shutil import copyfile
from multiprocessing import Manager
from multiprocessing.pool import Pool

from helper.history import HistoryManager

import searchHelper.search_parameter as SP
from searchHelper.fashion_MNIST_evaluation import get_evaluator
from searchHelper.rule_generation import generate_missing_rules, calculate_num_of_rules

if __name__ == '__main__':
    if not path.exists(SP.DIRECTORY_PATH):  # TODO
        makedirs(SP.DIRECTORY_PATH)

    copyfile('searchHelper/search_parameter.py', SP.PARAMETER_PATH)

    pool = Pool(processes=SP.NUM_OF_CORES_USED)
    manager = Manager()
    queue = manager.Queue()

    all_rules = generate_missing_rules()
    num_rules = calculate_num_of_rules()

    with HistoryManager(queue, SP.LOG_PATH, num_rules) as history_manager:
        print(f'Evaluate {num_rules} rules.')

        rule_evaluator = get_evaluator(queue)
        achieved_accuracies = pool.map(rule_evaluator, all_rules)

        best_accuracy = max(achieved_accuracies)
        index_list = range(num_rules)
        best_rule = all_rules[max(index_list, key=lambda x: achieved_accuracies[x])]
        print(f'Finished. Maximum achieved accuracy {best_accuracy} with {best_rule}.')
