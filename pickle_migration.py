import pickle
import sys
from searchHelper import history

sys.modules['history'] = history

path = 'logs/120719 W2P2/logs.pkl'

obj = []

with open(path, 'rb') as file:
    try:
        while True:
            obj += pickle.load(file)
    except EOFError:
        pass

del sys.modules['history']

with open(path, 'wb') as file:    
    pickle.dump(obj, file)
