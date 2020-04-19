import updateModel.model_parameter as MP


"""
An instance describes a possible weight-update and neuron-performance-update rule.
"""
class UpdateRule():  

    def __init__(self):
        self.performance_lr = 0
        self.weight_lr = 0

        self.performance_coefficients = [0] * MP.NUM_PERFORMANCE_SUMMANDS
        self.weight_coefficients = [0] * MP.NUM_WEIGHT_SUMMANDS

    """
    Returns the coefficient of the given variable.
    Works for both the weight- and the performance-update rule.
    """
    def __getitem__(self, variable):
        if variable in MP.PERFORMANCE_VARIABLES:
            index = MP.PERFORMANCE_VARIABLES.index(variable)
            return self.performance_coefficients[index]

        elif variable in MP.WEIGHT_VARIABLES:
            index = MP.WEIGHT_VARIABLES.index(variable)
            return self.weight_coefficients[index]

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    """
    Sets the coefficient of the given variable.
    Works for both the weight- and the performance-update rule.
    """
    def __setitem__(self, variable, coefficient):
        if variable in MP.PERFORMANCE_VARIABLES:
            index = MP.PERFORMANCE_VARIABLES.index(variable)
            self.performance_coefficients[index] = coefficient

        elif variable in MP.WEIGHT_VARIABLES:
            index = MP.WEIGHT_VARIABLES.index(variable)
            self.weight_coefficients[index] = coefficient

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    """
    Returns True iff the coefficient of the given variable is either 1 or -1.
    """
    def is_set(self, variable):
        if variable in MP.PERFORMANCE_VARIABLES:
            index = MP.PERFORMANCE_VARIABLES.index(variable)
            return 0 != self.performance_coefficients[index]

        elif variable in MP.WEIGHT_VARIABLES:
            index = MP.WEIGHT_VARIABLES.index(variable)
            return 0 != self.weight_coefficients[index]

        else:
            raise KeyError('Unknown variable name: ' + str(variable))

    def __str__(self):  # TODO
        string = 'p = '
        string += ' '.join([('+ ' if c > 0 else '- ') + var for (c, var) in 
                            zip(self.performance_coefficients, MP.PERFORMANCE_VARIABLES)
                            if c != 0])                            

        string += '; d_w = '
        string += ' '.join([('+ ' if c > 0 else '- ') + var for (c, var) in 
                            zip(self.weight_coefficients, MP.WEIGHT_VARIABLES)
                            if c != 0])

        string += "; (" + '{:.4f}'.format(self.performance_lr) + "," + \
                    '{:.4f}'.format(self.weight_lr) + ")"

        return string

    def coefficient_string(self):
        return ",".join(map(str, self.performance_coefficients + self.weight_coefficients))
