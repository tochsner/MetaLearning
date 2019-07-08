"""
Stores all the parameters in a centralized file.
"""

EXPERIMENT_NAME = '030719 W2P3'

"""
Parameters describing the general model used to update the weights.
"""
OUTPUT_FITNESS_FUNCTION = lambda y, y_target: y - y_target

# max number of chosen summands if a rule is randomly generated
MAX_NUM_OF_SUMMANDS = 4

# number of chosen summands if the all possible rules are tested
NUM_OF_CHOSEN_WEIGHT_SUMMANDS = 2
NUM_OF_CHOSEN_PERFORMANCE_SUMMANDS = 3

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

NUM_OF_CORES = 2

