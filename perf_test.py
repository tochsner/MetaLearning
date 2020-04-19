import numpy as np
from timeit import default_timer as timer
from benchmarkHelper.run_benchmark import evaluate
from update_rule import UpdateRule
from data.fashion_MNIST import load_data, prepare_data_for_tooc
from general_NN import GeneralNeuralNetwork
import benchmarkHelper.benchmark_parameter as BCN
import model_parameter as MCN

np.random.seed(0)

rule = UpdateRule()

"""
rule.set('p1*p2', -1)
rule.set('p1*y2', 1)
rule.set('p2*y1', 1)
rule.set('y1*y2', 1)
"""
rule.set('p2*y2', 1)

rule.performance_lr = 0.5
rule.weight_lr = 1

NN = GeneralNeuralNetwork((784, 100, 10), lambda x:x, lambda x, y:x - y, rule)

start = timer()
for i in range(50000):
    NN.adapt_weights()
end = timer()

print(end - start)

"""
s1 = 1000
s2 = 100

n1 = np.random.rand(s1, 1)
n2 = np.random.rand(s2, 1)

nn1 = np.ravel(n1)
nn2 = np.ravel(n2)

def calc1():
    return n2.T * n1

def calc2():
    return np.outer(n1, n2)
    
def calc3():
    return np.dot(n1, n2.T)

def calc4():
    return nn1[np.newaxis].T * nn2

def calc5():
    return np.outer(nn1, nn2)
    
def calc6():
    return np.dot(nn1[np.newaxis].T, nn2[np.newaxis])


print(calc1())
print(calc2())
print(calc3())
print(calc4())
print(calc5())
print(calc6())

def run(func):
    start = timer()
    for i in range(10000):
        func()
    end = timer()
    print(end - start)

run(calc1)
run(calc2)
run(calc3)
run(calc4)
run(calc5)
run(calc6)
"""