import multiprocessing
import time
import sys
import random
import pebble

from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition, Categorical, Integer
import subprocess

from gbd_core.api import GBD
from concurrent.futures import as_completed
from confspace_toplevel import *



instancegroup = int(sys.argv[1]) #index of the instance family group to get from gbd (family groups are defined in instance_families.txt)

timeout = int(sys.argv[2]) #timeout for a single instance

kinstances = int(sys.argv[3]) #take k instances out of training set each run

paralleldeg = int(sys.argv[4]) #how many parallel threads are going to be scheduled

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

def runKissat(args):
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
        print("TIMEOUT: Instance " + args[1] + " has reached timeout, punishment {} seconds".format(2 * timeout), flush=True)
        return 2 * timeout


def run(config, seed: int = 0): #-> float
    print("Config {}".format(config))
    totaltime = 0
    called = 0

    arglist = []
    for file in inst[:kinstances]:
        args = ("/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/kissat/kissat_satcomp24", 
                file, 
                "--time=" + str(timeout),
                "-q",
                "-n")

        args = args + (config,)

        arglist.append(args)

    print("Number args:" + str(len(arglist)))
    
    print("Starting pool with {} threads and {} instances".format(paralleldeg, kinstances))
    with pebble.ProcessPool(max_workers=paralleldeg) as p:
        futures = [p.schedule(runKissat, (args,)) for args in arglist]
        for f in as_completed(futures):
            try:    
                totaltime += f.result()
                called += 1
                print("Result added to total", flush=True)
            except Exception as e:
                print(e)
    print(called)
    totaltime += (kinstances - called) * timeout

        
    return totaltime

random.seed(52)
inst = getinstances()
random.shuffle(inst)

space = get_kissat2024_confspace()

configs = []

for key in list(space.keys()):
    if key == "reorder":
        configs.append("--reorder=1")
        configs.append("--reorder=0")
    else:
        configs.append("--{}=0".format(key))

        if space.get(key).legal_value(2):
            configs.append("--{}=2".format(key))

for conf in configs:
    print("{}:{} finished with score {}".format(instancegroup, conf, run(conf)))