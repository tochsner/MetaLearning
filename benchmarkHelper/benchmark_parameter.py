"""
Stores all the parameters for the benchmark evaluation in a centralized file.
"""

LOGS_PATH = 'logs'
EXPERIMENT_NAME = '220819 Elite'
DIRECTORY_PATH = 'logs/' + EXPERIMENT_NAME
LOG_PATH = DIRECTORY_PATH + '/logs.pkl'
PARAMETER_PATH = DIRECTORY_PATH + '/parameter.py'

ELITE_PATH = 'logs/elite.pk'

# the accuracy threshold for a rule to be considered as elite
ELITE_ACCURACY_THRESHOLD = 0.6

# number of variations (regarding weight rule lr and performance rule lr)
# per rule
NUM_VARIATIONS_PER_RULE = 10

"""
Parameters describing the neural networks and hyperparameters for training.
"""
NETWORK_SIZE = (784, 40, 20, 10)
EPOCHS = 10

NUM_OF_CORES = 2
