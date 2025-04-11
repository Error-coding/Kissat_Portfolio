import time
import os

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
import subprocess




def train(config: Configuration, instance: str, seed: int = 0): #-> float:
	totaltime = 0
	args = ("../kissat/kissat", 
		instance,
		"--target=" + str(config["target"]),
		"--restartint=" + str(config["restartint"]),
		"--probeint=" + str(config["probeint"]),
		"--vivify=" +str(config["vivify"]),
		"--ands=" + str(config["ands"]),
		"--time=1000",
		"-n",
		"-q")
	start = time.time()
	popen = subprocess.Popen(args)
	popen.wait()
	end = time.time()
	totaltime += end - start
	return totaltime





configspace = ConfigurationSpace({"target": (0, 2), "restartint": (1,10000), "probeint": (2, 1000), "vivify": ["true", "false"], "ands": ["true","false"]})

instancelist = list(map(lambda x : "../instances/train/coloring/" + x, os.listdir("../instances/train/coloring")))


# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=50, objectives="runtime", instances=instancelist)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
print(incumbent)
