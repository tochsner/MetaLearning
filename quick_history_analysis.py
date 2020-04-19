import pickle
from helper import history
from searchHelper import toy_evaluation

name = '180420 W1P1'
path = 'logs/' + name + '/logs.pkl'

history = []

def print_best(threshold):
    best = [item for item in history if sum(item.accuracy_history) / len(item.accuracy_history) > threshold]

    best.sort(key=lambda item: sum(item.accuracy_history) / len(item.accuracy_history), reverse=True)

    for item in best:
        #print(toy_evaluation.evaluate(item.rule), max(item.accuracy_history))
        print('{:.2f}'.format(sum(item.accuracy_history) / len(item.accuracy_history)),
                '; Used Rule:',
                item.rule)

with open(path, 'rb') as file:
    try:
        while True:
            history += pickle.load(file)
    except EOFError:
        pass

print('Summary of', name)
print('Number of Items:', len(history))

print_best(0.0)