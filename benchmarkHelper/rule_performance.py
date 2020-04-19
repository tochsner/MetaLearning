class RulePerformance:

    def __init__(self, rule):
        self.rule = rule
        self.classification_performances = {}
        self.regression_performances = {}

    def add_classification_performance(self, performance):
        self.classification_performances[performance.benchmark_name] = performance

    def add_regression_performance(self, performance):
        self.regression_performances[performance.benchmark_name] = performance

    def average_classification_performance(self):
        return sum([perf.best_accuracy() for perf in self.classification_performances.items()]) / \
                len(self.classification_performances.items())

    def average_regression_performance(self):
        return sum([perf.best_accuracy() for perf in self.regression_performances.items()]) / \
                len(self.regression_performances.items())
    

class BenchmarkPerformance:

    def __init__(self, benchmark_name):        
        self.benchmark_name = benchmark_name
        self.results = []

    def best_accuracy(self):
        return max([result.test_accuracy for result in self.results])

    def add_benchmark_result(self, result):
        self.results += [result]


class BenchmarkResult:

    def __init__(self, rule, test_accuracy, train_accuracy, epochs, network_size):
        self.rule = rule
        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.epochs = epochs
        self.network_size = network_size
