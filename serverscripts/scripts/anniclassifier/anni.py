import multiprocessing
import time
import sys
import random
import pebble
import pickle as pkl

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess

from gbd_core.api import GBD
from concurrent.futures import as_completed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv




#instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = 1800
classifier = "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/anniclassifier/modelk9.pkl"
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


def get_available_features():
    with GBD([ "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/base.db" ]) as gbd:
        feat = gbd.get_features('base')
        feat.remove("base_features_runtime")
        feat.remove("ccs")
        feat.remove("bytes")
        return feat

def classify_instances():
    with open(classifier, 'rb') as model_file:
        loaded_model = pkl.load(model_file)
        with GBD(['/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/base.db', '/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/meta.db', "/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/instances/database/instances.db"]) as gbd:
            features = get_available_features()
            df = gbd.query('track=anni_2022 and track!=main_2023 and track!=main_2024 and minisat1m!=yes', resolve = features + ["instances:local"])
            X = df[features]
            y_pred = loaded_model.predict(X)
            classifications = list(zip(df["local"], y_pred))
            return classifications

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
        return 2 * timeout
    
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
        print("Instance " + args[1] +  " with config {} finished after {} seconds".format(config_str, end-start), flush=True)
        return end - start
    else:
        print("TIMEOUT: Instance " + args[1] + " with config {} has reached timeout, punishment {} seconds".format(config_str, 2 * timeout), flush=True)
        return 2 * timeout


def train(seed: int = 0): #-> float
    totaltime = 0
    called = 0
    classified = classify_instances()
    arglist = []



    for file, predclass in classified:
        config_str = ""
        args = ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/kissat/kissat_satcomp24", 
                file, 
                "--time=" + str(timeout),
                "-q",
                "-n")
        #with open("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/anniclassifier/famcluster10.txt", 'r') as f:
        #    lines = f.readlines()
        #    config = {}
        #    if predclass >= len(lines):
        #        print(f"Predicted class {predclass} is out of range, skipping this instance.")
        #        continue
        #    line = lines[predclass].strip()
        if "Default" in str(predclass):
            config_str = "Default"
            config = {}
        else:
            config_str= predclass
            config = parse_dict_string(config_str)
        for key in config:
            arg = "--" + key + "=" + str(config[key])
            args = args + (arg,)
        arglist.append(args + (config_str,))
    
    with pebble.ProcessPool(max_workers=paralleldeg) as p:
        futures = [p.schedule(runKissat, (args,)) for args in arglist]
        for f in as_completed(futures):
            try:    
                totaltime += f.result()
            except Exception as e:
                print(e)
    print(called)
        
    return totaltime


print(len(inst))


print("Terminated after {} seconds".format(train()))