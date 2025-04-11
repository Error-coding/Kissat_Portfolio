import multiprocessing
import time
import sys
import random
import pebble

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess
import csv


from gbd_core.api import GBD
from concurrent.futures import as_completed



#instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = 1800

#kinstances = int(sys.argv[3]) #take k instances out of training set each run

 #how many times the train function is going to be called

paralleldeg = 32 #how many parallel threads are going to be scheduled

inst = []

def parse_dict_string(dict_string):
    dict_string = dict_string.strip('{}')
    items = dict_string.split(', ')
    parsed_dict = {}
    for item in items:
        key, value = item.split(': ')
        key = key.strip("'")
        value = int(value)
        parsed_dict[key] = value
    return parsed_dict

# Example usage
dict_string = "{'backbone': 0, 'bump': 1, 'chrono': 1, 'congruence': 1, 'eliminate': 1, 'extract': 0, 'factor': 1, 'fastel': 0, 'forward': 1, 'lucky': 0, 'phase': 1, 'phasesaving': 1, 'preprocess': 0, 'probe': 1, 'randec': 1, 'reluctant': 1, 'reorder': 0, 'rephase': 0, 'restart': 1, 'stable': 0, 'substitute': 0, 'sweep': 1, 'target': 0, 'transitive': 0, 'vivify': 1, 'warmup': 1}"


parsed_dict = parse_dict_string(dict_string)
print(parsed_dict)

def getinstances():
    instlist = []
    with GBD (["/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/meta.db" , "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/instances.db"]) as gbd:
        
        feat = ["instances:local"]
        df = gbd.query ( "track=anni_2022 and track!=main_2023 and track!=main_2024 and minisat1m!=yes", resolve = feat)
        #print(df["local"].tolist())
        instlist = df["local"].tolist()

    return instlist

def runKissat(args):
    def check_resolved_csv(filename, config_str):
        try:
            with open('/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/records/resolved.csv', mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['key'] == filename and row['configuration'] == config_str:
                        time = float(row['time'])
                        if time >= timeout:
                            return 2 * timeout
                        else:
                            return time
        except FileNotFoundError:
            pass
        return None

    filename = args[1].split("/")[-1].split("-")[0]
    config_str = args[-1]
    resolved_time = check_resolved_csv(filename, config_str)
    if resolved_time is not None:
        print(f"Instance {filename} with config {config_str} already resolved in {resolved_time} seconds", flush=True)
        return resolved_time


    start = time.time()
    output = None
    try:
    # Run the subprocess without `check=True`
        print(args[:-1], flush=True)
        output = subprocess.run(args[:-1], capture_output=True)

    # Check the returncode manually and handle 10 and 20 exit codes
        if output.returncode not in {0, 10, 20}:
            raise subprocess.CalledProcessError(output.returncode, args[:-1], output=output.stdout, stderr=output.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error in running kissat: {e}", flush=True)
        return 2* timeout
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
            #print(str(instancegroup) + ": Solved", flush=True)
            status = True
            break
        elif line == r's UNKNOWN':
            #print(str(instancegroup) + ": Timeout", flush=True)
            status = False
            break
        else:
            print("Solver failed", flush=True)
            status = False

    with open("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/records/resolved.csv", 'a', newline='') as csvfile:
        fieldnames = ['key', 'time', 'configuration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        #print("Writing to savefile: {}, {}".format(filename, config_str))
        writer.writerow({'key': filename, 'time': end - start, 'configuration': config_str})

    if(status):
        print("Instance " + args[1] + "finished after {} seconds".format(end-start), flush=True)
        return end - start
    else:
        print("TIMEOUT: Instance " + args[1] + " has reached timeout, punishment {} seconds".format(2 * timeout), flush=True)
        return 2 * timeout


def train( config,seed: int = 0): #-> float
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
        arglist.append(args + (dict_string,))
    
    with pebble.ProcessPool(max_workers=paralleldeg) as p:
        futures = [p.schedule(runKissat, (args,)) for args in arglist]
        for f in as_completed(futures):
            try:    
                totaltime += f.result()
            except Exception as e:
                print(e)
    print(called)
        
    return totaltime

inst = getinstances()

print(len(inst))

# anni1 "{'backbone': 0, 'bump': 1, 'chrono': 1, 'congruence': 0, 'eliminate': 1, 'extract': 1, 'factor': 0, 'fastel': 1, 'forward': 1, 'lucky': 0, 'phase': 0, 'phasesaving': 0, 'preprocess': 0, 'probe': 1, 'randec': 1, 'reluctant': 0, 'reorder': 1, 'rephase': 0, 'restart': 1, 'stable': 0, 'substitute': 1, 'sweep': 1, 'target': 1, 'transitive': 0, 'vivify': 1, 'warmup': 0}"
args = parse_dict_string(dict_string)

print("Terminated after {} seconds".format(train(args)))