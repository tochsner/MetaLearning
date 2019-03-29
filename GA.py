"""
Implementation of a Vanilla Genetic Algorithm.
"""

import random
from selectors import BaseSelector

class BaseGA:
    def initialize(self):
        pass
    
    def perform_iteration(self):
        pass

    def get_best_fitness(self):
        pass

    def get_average_fitness(self):
        pass


class GAParameter:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate


class VanillaGA(BaseGA):    
    def __init__(self, parameter, solution_generator, fitness_function, mutation_function, selector):
        self.parameter = parameter
        self.solution_generator = solution_generator
        self.fitness_function = fitness_function
        self.mutation_function = mutation_function
        self.selector = selector

        self.population = []
        self.fitness = {}
        
    def initialize(self):
        for i in range(self.parameter.population_size):
            self.population.append(self.solution_generator())            
        
    def perform_iteration(self):              
        self.evaluate_fitness()            
        self.population = self.selector.generate_new_population(self.population, self.fitness)                
        self.mutate_population()
            
    def evaluate_fitness(self):
        self.fitness = {}
        for individual in self.population:
            self.fitness[individual] = self.fitness_function(individual)
    
    def mutate_population(self):
        for individual in self.population:
            if random.random() < self.parameter.mutation_rate:
                self.mutation_function(individual)

    def get_best_fitness(self):
        return max(self.fitness.values())

    def get_average_fitness(self):        
        return sum(self.fitness.values()) / len(self.fitness)
