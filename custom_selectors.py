import itertools
import numpy as np
import copy

class BaseSelector:
    def generate_new_population(self, old_population, fitness_of):
        pass


"""
Creates a new population by tournament selection.
To reduce noise introduced by random selection of pairs, 
two permutations of the whole population are created and
the fitter solution at each index taken.
"""
class TournamentSelector(BaseException):
    def generate_new_population(self, old_population, fitness_of):
        new_population = []
        
        population_size = len(old_population)

        permutation = np.random.permutation(population_size)

        for i, j in enumerate(permutation):
            candidate1 = old_population[i]
            candidate2 = old_population[j]            

            if fitness_of[candidate1] >= fitness_of[candidate2]:
                new_population.append(copy.deepcopy(candidate1))
            else:
                new_population.append(copy.deepcopy(candidate2))

        for s in old_population:
            del s

        return new_population