"""
Stores all the parameters for the exhaustive search in a centralized file.
"""

EXPERIMENT_NAME = '190420 W2P1'
DIRECTORY_PATH = 'logs/' + EXPERIMENT_NAME
LOG_PATH = DIRECTORY_PATH + '/logs.pkl'
PARAMETER_PATH = DIRECTORY_PATH + '/parameter.py'

NUM_OF_CORES_USED = 6

"""
Parameters describing the rules tested.
"""
NUM_OF_CHOSEN_WEIGHT_SUMMANDS = 2
NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS = 1

# number of variations (regarding weight rule lr and performance rule lr)
# per rule if all possible rules are tested
NUM_VARIATIONS_PER_RULE = 4

"""
Parameters describing the neural networks and hyperparameters for training.
"""
NETWORK_SIZE = (784, 40, 20, 10)

MAX_EPOCHS = 1
# one batch is the minimum number of samples processed before early stopping
BATCH_SIZE = 1000
# the accuracy threshold for early stopping
ACCURACY_THRESHOLD = 0.15
