from random import random, randint, choice
from itertools import combinations, product
import parameter as CN

"""
An instance describes a possible weight-update and neuron-performance-update rule.
"""
class UpdateRule():  

    # number of summands of the weight-update rule
    NUM_WEIGHT_SUMMANDS = 10
    # number of summands of the performance-update rule
    NUM_PERFORMANCE_SUMMANDS = 7

    # variables of the weight-update-rule
    PERFORMANCE_VARIABLES = ['y',
                        'p_out',
                        'y^2',
                        'p_out^2',
                        'y*p_out',
                        'y^2*p_out',
                        'y*p_out^2']
    # variables of the performance-update rule
    WEIGHT_VARIABLES = ['p1',
                             'p2',
                             'y1',
                             'y2',
                             'p1*y1',
                             'p2*y2',
                             'p1*p2',
                             'p1*y2',
                             'p2*y1',
                             'y1*y2']

    def __init__(self):
        self.performance_lr = 0
        self.weight_lr = 0

        self.performance_coefficients = [0] * self.NUM_PERFORMANCE_SUMMANDS
        self.weight_coefficients = [0] * self.NUM_WEIGHT_SUMMANDS      

    """
    Generates a random update rule.

    lr lies in [0,1] and is calculated by lr = x**2, where x is uniformly drawn from [0,1].
    The coefficients are either 0, 1 or -1.
    """
    @classmethod
    def generate_random_rule(cls):
        rule = UpdateRule()

        rule.performance_lr = random() ** 2
        rule.weight_lr = random() ** 2

        num_chosen_performance_summands = randint(1, CN.MAX_NUM_OF_SUMMANDS)
        num_chosen_weight_summands = randint(1, CN.MAX_NUM_OF_SUMMANDS)

        for i in range(num_chosen_performance_summands):
            index = randint(0, cls.NUM_PERFORMANCE_SUMMANDS - 1)
            coefficient = choice([-1, 1])
            rule.performance_coefficients[index] = coefficient

        for i in range(num_chosen_weight_summands):
            index = randint(0, cls.NUM_WEIGHT_SUMMANDS - 1)
            coefficient = choice([-1, 1])
            rule.weight_coefficients[index] = coefficient

        return rule

    """
    Generates all possible rules according to the parameter file.

    lr lies in [0,1] and is calculated by lr = x**2, where x is uniformly drawn from [0,1].
    The coefficients are either 0, 1 or -1.
    """
    @classmethod
    def generate_all_rules(cls):             
        for chosen_perf_summands in combinations(range(cls.NUM_PERFORMANCE_SUMMANDS)
                                                , CN.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS):            

            for chosen_weight_summands in combinations(range(cls.NUM_WEIGHT_SUMMANDS)
                                                    , CN.NUM_OF_CHOSEN_WEIGHT_SUMMANDS):                

                for perf_coefficients in product([-1, 1], repeat=CN.NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS):

                    for weight_coefficients in product([-1, 1], repeat=CN.NUM_OF_CHOSEN_WEIGHT_SUMMANDS):

                        for v in range(CN.NUM_VARIATIONS_PER_RULE):
                            rule = UpdateRule()
                    
                            rule.performance_lr = 10 ** (-random() * 2)
                            rule.weight_lr = 10 ** (-random() * 2)

                            for s, c in zip(chosen_perf_summands, perf_coefficients):
                                rule.performance_coefficients[s] = c

                            for s, c in zip(chosen_weight_summands, weight_coefficients):
                                rule.weight_coefficients[s] = c

                            yield rule

    """
    Returns the coefficient of the given variable.
    Works for both the weight- and the performance-update rule.
    """
    def get(self, variable):
        if variable in self.PERFORMANCE_VARIABLES:
            index = self.PERFORMANCE_VARIABLES.index(variable)
            return self.performance_coefficients[index]

        elif variable in self.WEIGHT_VARIABLES:
            index = self.WEIGHT_VARIABLES.index(variable)
            return self.weight_coefficients[index]

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    """
    Sets the coefficient of the given variable.
    Works for both the weight- and the performance-update rule.
    """
    def set(self, variable, coefficient):
        if variable in self.PERFORMANCE_VARIABLES:
            index = self.PERFORMANCE_VARIABLES.index(variable)
            self.performance_coefficients[index] = coefficient

        elif variable in self.WEIGHT_VARIABLES:
            index = self.WEIGHT_VARIABLES.index(variable)
            self.weight_coefficients[index] = coefficient

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    """
    Returns True iff the coefficient of the given variable is either 1 or -1.
    """
    def is_set(self, variable):
        if variable in self.PERFORMANCE_VARIABLES:
            index = self.PERFORMANCE_VARIABLES.index(variable)
            return (0 != self.performance_coefficients[index])

        elif variable in self.WEIGHT_VARIABLES:
            index = self.WEIGHT_VARIABLES.index(variable)
            return (0 != self.weight_coefficients[index])

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    def __str__(self):
        string = 'p = '
        string += ' '.join([('+ ' if c > 0 else '- ') + var for (c, var) in 
                            zip(self.performance_coefficients, self.PERFORMANCE_VARIABLES)
                            if c != 0])                            

        string += '; d_w = '
        string += ' '.join([('+ ' if c > 0 else '- ') + var for (c, var) in 
                            zip(self.weight_coefficients, self.WEIGHT_VARIABLES)
                            if c != 0])

        string += "; (" + '{:.4f}'.format(self.performance_lr) + "," + \
                    '{:.4f}'.format(self.weight_lr) + ")"

        return string