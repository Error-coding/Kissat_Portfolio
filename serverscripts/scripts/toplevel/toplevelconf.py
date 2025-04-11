import time
import sys
import random

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess

from gbd_core.api import GBD
from confspace_toplevel import *
    
seed = 31

instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = int(sys.argv[2]) #timeout for a single instance

kinstances = int(sys.argv[3]) #take k instances out of training set each run

ntrials = int(sys.argv[4]) #how many times the train function is going to be called

inst = []

def getinstances():
    print("Getting instances", flush=True)
    f=open("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances_families.txt")
    lines=f.readlines()
    fams = lines[instancegroup].split()[1].split(",")

    famstring = "(family=" + fams[0]

    for i in range(1, len(fams)):
        famstring += " or family=" + fams [i]
    famstring += ")"


    instlist = []
    with GBD (["/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/meta.db" , "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/instances.db"]) as gbd:
        
        feat = ["instances:local"]
        df = gbd.query ( "(track=main_2023 or track=main_2024) and " + famstring + " and minisat1m!=yes", resolve = feat)
        print(df["local"].tolist())
        instlist = df["local"].tolist()

    return list(map(lambda x : ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/train/" + x.split("/")[-1])[:-3], instlist))

# function to
def train(config: Configuration, seed: int = 0): #-> float:
    totaltime = 0
    
    print("---Testing configuration---", flush=True)
    print(config, flush=True)
    print("On {} instances".format(min(len(inst), kinstances)), flush=True)
    print("Worst case runtime for this iteration is {} seconds".format(min(len(inst), kinstances) * timeout), flush=True)

    for file in inst[:kinstances]:
        print("Running configuration on file " + file, flush=True)

        args = ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/kissat/kissat_satcomp24", 
            file, 
            "--time=" + str(timeout),
            "-q",
            "-n")

        for key in config:
            arg = "--" + key + "=" + str(config[key])
            args = args + (arg,)
        
        start = time.time()
        output= None
        try:
            output = subprocess.run(args, capture_output=True)
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)

        end = time.time()

        if output is None:
            print("No output from kissat, returning early", flush=True)
            return 2 * timeout * kinstances

        outputstr = output.stdout.decode()

        status = False
        for line in outputstr.splitlines():
            line = line.strip()
            if (line == r's SATISFIABLE') or (line == r's UNSATISFIABLE'):
                print("Solved", flush=True)
                status = True
                break
            elif line == r's UNKNOWN':
                print("Timeout", flush=True)
                status = False
                break

        if(status):
            print("Instance " + file + "finished after {} seconds".format(end-start), flush=True)
            totaltime += end - start
        else:
            totaltime += 2 * timeout
    return totaltime


inst = getinstances()

print("Shuffling instances with seed {}".format(seed))
random.seed(seed)
random.shuffle(inst)

print("Setting up config")

configspace = get_kissat2024_confspace()


# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=ntrials, objectives="runtime", output_directory = "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/outputs/toplevel/" + str(instancegroup))

# Use SMAC to find the best configuration/hyperparameters

print("Starting smac")
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
print(incumbent)
