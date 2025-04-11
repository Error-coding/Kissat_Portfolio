from confspace import *
import multiprocessing
import time
import sys
import random
import pebble

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess

from gbd_core.api import GBD
from concurrent.futures import as_completed


instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = int(sys.argv[2]) #timeout for a single instance

ntrials = int(sys.argv[3]) #how many times the train function is going to be called

paralleldeg = int(sys.argv[4]) #how many parallel threads are going to be scheduled

inst = []

def getinstances(hashes):
    print("Getting instances", flush=True)


    instlist = []
    with GBD (["/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/meta.db" , "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/instances.db"]) as gbd:
        for i in hashes:
            feat = ["instances:local"]
            df = gbd.query ( "hash=\"{}\"".format(i), resolve = feat)
            print(df["local"].tolist())
            instlist.extend(df["local"].tolist())

    return list(map(lambda x : ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/train/" + x.split("/")[-1])[:-3], instlist))

def runKissat(args):
    print("Started kissat")
    start = time.time()
    output = None
    try:
    # Run the subprocess without `check=True`
        output = subprocess.run(args, capture_output=True)

    # Check the returncode manually and handle 10 and 20 exit codes
        if output.returncode in {10, 20}:
            print(f"Process exited with expected code {output.returncode}, no error: {output.stderr.decode()}", flush=True)
        elif output.returncode != 0:
        # If it's some other non-zero exit code, raise an error
            raise subprocess.CalledProcessError(output.returncode, args, output=output.stdout, stderr=output.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error in running kissat: {e}", flush=True)
    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
    
    end = time.time()

    if output is None:
        print("No output from kissat, failing after {} seconds".format(end-start), flush=True)
        return 2 * timeout
    
    outputstr = output.stdout.decode()

    status = True
    for line in outputstr.splitlines():
        line = line.strip()
        if (line == r's SATISFIABLE') or (line == r's UNSATISFIABLE'):
            print(str(instancegroup) + ": Solved", flush=True)
            status = True
            break
        elif line == r's UNKNOWN':
            print(str(instancegroup) + ": Timeout", flush=True)
            status = False
            break
        else:
            print("Solver failed", flush=True)
            status = False

    if(status):
        print("SOLVED: Instance " + args[1] + "was solved after {} seconds".format(end-start), flush=True)
        return end - start
    else:
        print("TIMEOUT: Instance " + args[1] + " has reached timeout, punishment {} seconds".format(2 * 5000), flush=True)
        return 2 * timeout
    
def train(config: Configuration, seed: int = 0): #-> float
    totaltime = 0
    called = 0

    arglist = []
    for file in inst:
        args = ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/kissat/kissat_satcomp24", 
                file, 
                "--time=" + str(timeout),
                "-q",
                "-n")

        for key in config:
            arg = "--" + key + "=" + str(config[key])
            args = args + (arg,)
        arglist.append(args)

    print("Number args:" + str(len(arglist)))
    
    print("Starting pool with {} threads and {} instances".format(paralleldeg, len(inst)))
    with pebble.ProcessPool(max_workers=paralleldeg) as p:
        futures = [p.schedule(runKissat, (args,)) for args in arglist]
        for f in as_completed(futures):
            try:    
                totaltime += f.result()
                called += 1
                print("Result added to total", flush=True)
            except Exception as e:
                print(e)
        
    return totaltime


def parse_string_to_dict(input_string):
    # Remove the curly braces and split the string by commas
    items = input_string.strip('{}').split(',')
    
    # Initialize an empty dictionary
    result_dict = {}
    
    # Iterate over the items and split by colon to form key-value pairs
    for item in items:
        key, value = item.split(':')
        result_dict[key.strip("'")] = int(value)
    
    return result_dict


f=open("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/toplevelsplit.txt")
line=f.readlines()[instancegroup]
instances = line.split(" ")[1:]
configin = parse_string_to_dict(line.split(" ")[0])

inst = getinstances(instances)
confspace = []
if configin != "Default":
    confspace = get_options(configin)
else:
    confspace = getall()

print(confspace, flush=True)

# Scenario object specifying the optimization environment
scenario = Scenario(confspace, deterministic=True, n_trials=ntrials, output_directory = "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/outputs/extend/liskov/" + str(instancegroup))

# Use SMAC to find the best configuration/hyperparameters

print("Starting smac")
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
print(incumbent)

