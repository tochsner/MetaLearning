"""
Runs an in-depth evaluation of the best rules found in exhaustive search.
As evaluation, the FashionMNIST dataset is used.

The search runs in parallel, with the results saved in a shared queue
and stored periodically on disc by the history object.
"""

from multiprocessing.pool import Pool
from multiprocessing import Manager
from os import makedirs, path
from shutil import copyfile

from benchmarkHelper.fashion_MNIST_evaluation import evaluate
import benchmarkHelper.benchmark_parameter as BCN
from benchmarkHelper.get_elite_rules import get_elite_rules, get_missing_elite_rules
from helper.history import HistoryManager

if __name__ == '__main__':
    if not path.exists(BCN.DIRECTORY_PATH):
        makedirs(BCN.DIRECTORY_PATH)

    copyfile('searchHelper/search_parameter.py', BCN.PARAMETER_PATH)
    
    manager = Manager()
    queue = manager.Queue()
    
    all_rules = list(get_missing_elite_rules())
    all_solutions = [(queue, rule) for rule in all_rules]

    history_manager = HistoryManager(queue, BCN.LOG_PATH, len(all_solutions))    

    print('Evaluate', len(all_rules), 'rules')

    pool = Pool(processes=BCN.NUM_OF_CORES)
    results = pool.map(evaluate, all_solutions)

    history_manager.end()

    index_list = range(len(results))    
    print('Finished. Maximum achieved accuracy ', max(results), 'with', 
            all_solutions[max(index_list, key=lambda x: results[x])][1])
