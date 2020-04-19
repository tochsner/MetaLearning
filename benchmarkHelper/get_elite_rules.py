from os import scandir
import pickle

from benchmarkHelper import benchmark_parameter as BCN
from searchHelper.rule_generation import generate_variation


def get_elite_rules():
	history = []

	for path in [f.path for f in scandir(BCN.LOGS_PATH) if f.is_dir()]:
		if 'Elite' in path:
			continue

		log_path = path + '\logs.pkl'

		with open(log_path, 'rb') as file:
			try:
				while True:
					history += pickle.load(file)
			except EOFError:
				pass

	elite = filter(lambda item: max(item.accuracy_history) > BCN.ELITE_ACCURACY_THRESHOLD,
						history)

	# add to dictionary to eliminate duplicate rules
	elite_rules = {item.rule.coefficient_string() : item.rule for item in elite}	

	variations = []

	for rule in elite_rules.values():
		for i in range(BCN.NUM_VARIATIONS_PER_RULE):
			variations += [generate_variation(rule)]
	
	return variations


def get_missing_elite_rules():
	all_rules = []

	with open(BCN.ELITE_PATH, 'rb') as file:
		try:
			while True:
				all_rules += pickle.load(file)
		except EOFError:
			pass

	tested_rules = []

	for path in [f.path for f in scandir(BCN.LOGS_PATH) if f.is_dir()]:
		if 'Elite' not in path:
			continue

		log_path = path + '\logs.pkl'

		try:
			with open(log_path, 'rb') as file:
				try:
					while True:
						tested_rules += pickle.load(file)
				except EOFError:
					pass
		except FileNotFoundError:
			pass

	tested_rule_strings = [rule.rule.coefficient_string() for rule in tested_rules]

	all_rules = filter(lambda rule: tested_rule_strings.count(rule.coefficient_string()) < BCN.NUM_VARIATIONS_PER_RULE, all_rules)

	variations = []

	for rule in all_rules:
		for i in range(BCN.NUM_VARIATIONS_PER_RULE):
			variations += [generate_variation(rule)]

	return variations
