from os import scandir
import pickle

from benchmarkHelper import benchmark_parameter as BCN

def save_elite():
	history = []

	for path in [f.path for f in scandir(BCN.LOGS_PATH) if f.is_dir()]:
		if 'Elite' in path:
			print(path)
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

	for rule in elite_rules.values():
		rule.performance_lr = 0
		rule.weight_lr = 0
		
	with open(BCN.ELITE_PATH, 'bw') as file:
		pickle.dump(list(elite_rules.values()), file)

	print(len(elite_rules.values()))

save_elite()