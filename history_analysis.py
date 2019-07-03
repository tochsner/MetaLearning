import pickle

name = '030719 W1P1'
path = 'logs/' + name + '/logs.pkl'

def print_best(threshold):
    best = [item for item in history if sum(item.accuracy_history) / len(item.accuracy_history) > threshold]

    best.sort(key=lambda item: sum(item.accuracy_history) / len(item.accuracy_history), reverse=True)

    for item in best:
       print('{:.2f}'.format(sum(item.accuracy_history) / len(item.accuracy_history)), 
                '; Used Rule:',
                str(item.rule))

history = []

with open(path, 'rb') as file:
    try:
        while True:
            history += pickle.load(file)
    except EOFError:
        pass

print('Summary of', name)
print('Number of Items:', len(history))

print_best(0.3)