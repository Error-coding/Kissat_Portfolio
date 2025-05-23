import time
import sys
import random

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess

from gbd_core.api import GBD
    


instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = int(sys.argv[2]) #timeout for a single instance

kinstances = int(sys.argv[3]) #take k instances out of training set each run

ntrials = int(sys.argv[4]) #how many times the train function is going to be called


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
def train(seed: int = 0): #-> float:
    totaltime = 0
    
    inst = getinstances()
    print(inst)
    #random.shuffle(inst)
    for file in inst[:kinstances]:
        args = ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/kissat/kissat_satcomp24", 
            file, 
            "--time=" + str(timeout),
            "-q",
            "-n")

        
        start = time.time()
        try:
            output = subprocess.run(args, capture_output=True)
        except:
            print("Solver failed")
            
        end = time.time()
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

        if(status):
            totaltime += end - start
        else:
            totaltime += 2 * timeout
    return totaltime



random.seed(31)

print("Finished after {} seconds".format(train()))