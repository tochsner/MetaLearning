from random import random
from itertools import combinations, product
import pickle
import os.path
from scipy.special import comb as ncr

import searchHelper.search_parameter as SP
import updateModel.model_parameter as MP
from updateModel.update_rule import UpdateRule


"""
Calculates the total number of update rules.
"""
def calculate_num_of_rules():
    possible_performance_summands = 2 ** SP.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS * \
                                    ncr(MP.NUM_PERFORMANCE_SUMMANDS, SP.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS)
    possible_weight_summands = 2 ** SP.NUM_OF_CHOSEN_WEIGHT_SUMMANDS * \
                               ncr(MP.NUM_WEIGHT_SUMMANDS, SP.NUM_OF_CHOSEN_WEIGHT_SUMMANDS)

    return int(SP.NUM_VARIATIONS_PER_RULE * possible_performance_summands * possible_weight_summands)


"""
The returned lr lies in [0,1] and is calculated by lr = x * 10**(-y), where x and y 
are uniformly drawn from [0,1]
"""
def generate_random_lr():
    return random() * 10**(-random())


"""
Returns a rule with the same coefficients as the given rule but randomly chosen lr's.
"""
def generate_variation(rule):
    new_rule = UpdateRule()
    
    new_rule.performance_coefficients = rule.performance_coefficients
    new_rule.weight_coefficients = rule.weight_coefficients

    new_rule.performance_lr = generate_random_lr()
    new_rule.weight_lr = generate_random_lr()

    return new_rule


"""
Yields all possible rules according to the parameter file in a generator-wise manner.

lr are generated according to generate_random_lr().
The coefficients are either 0, 1 or -1.
"""
def generate_all_rules():
    for chosen_perf_summands in combinations(range(MP.NUM_PERFORMANCE_SUMMANDS)
                                                , SP.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS):

        for chosen_weight_summands in combinations(range(MP.NUM_WEIGHT_SUMMANDS)
                                                    , SP.NUM_OF_CHOSEN_WEIGHT_SUMMANDS):

            for perf_coefficients in product([-1, 1], repeat=SP.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS):

                for weight_coefficients in product([-1, 1], repeat=SP.NUM_OF_CHOSEN_WEIGHT_SUMMANDS):

                    for _ in range(SP.NUM_VARIATIONS_PER_RULE):
                        rule = UpdateRule()
                
                        rule.performance_lr = generate_random_lr()
                        rule.weight_lr = generate_random_lr()

                        for s, c in zip(chosen_perf_summands, perf_coefficients):
                            rule.performance_coefficients[s] = c

                        for s, c in zip(chosen_weight_summands, weight_coefficients):
                            rule.weight_coefficients[s] = c

                        yield rule


"""
Yields all possible rules which don't have a corresponding history_item, i.e. which
haven't been evaluated before.

lr lies in [0,1] and is calculated by lr = x**2, where x is uniformly drawn from [0,1].
The coefficients are either 0, 1 or -1.
"""
def generate_missing_rules():
    history_items = []

    if not os.path.isfile(SP.LOG_PATH):
        for rule in generate_all_rules():
            yield rule

        return

    with open(SP.LOG_PATH, 'rb') as file:
        try:
            while True:
                history_items += pickle.load(file)
        except EOFError:
            pass

    # set of all rules, ignoring duplicates due to variations of lr's
    existing_rules = {item.rule.coefficient_string() for item in history_items}

    for rule in generate_all_rules():
        if not rule.coefficient_string() in existing_rules:
            yield rule
