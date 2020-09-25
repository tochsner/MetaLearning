from benchmarkHelper.run_benchmark import evaluate
from update_rule import UpdateRule
from data.fashion_MNIST import load_data, prepare_data_for_tooc
from general_NN import GeneralNeuralNetwork
import benchmarkHelper.benchmark_parameter as BCN
import model_parameter as MCN

rule = UpdateRule()

rule.set('y*p_out', 1)
rule.set('y^2*p_out', -1)

rule.set('p2*y1', -1)

rule.performance_lr = 0.5
rule.weight_lr = 1

data = load_data()
samples = prepare_data_for_tooc(data)

evaluate(rule, 'Fashion_MNIST', samples)
