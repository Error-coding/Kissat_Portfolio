import json
from gbd_core.api import GBD

import pandas as pd


class SplitNode:
    def __init__(self, hashes=None, config=None, split_var=None, split_point=None, left=None, right=None):
        self.hashes = hashes
        self.config = config
        self.split_var = split_var
        self.split_point = split_point
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(hashes={self.hashes}, config={self.config})"
        else:
            return f"Node(split_var={self.split_var}, split_point={self.split_point}, left={self.left}, right={self.right})"

def getFeat(inst):
    hashstring = "hash=\"{}\"".format(inst[0])
    for i in range(len(inst)):
        hashstring += "or hash=\"{}\"".format(inst[i])
    


    with GBD (["/home/raphael-zipperer/Uni/BA/database/meta.db", "/home/raphael-zipperer/Uni/BA/database/base.db"]) as gbd:
        torem = ["bytes", "ccs", "base_features_runtime"]
        feat = gbd . get_features ("base")
        for i in torem:    
            feat.remove(i)
        df = gbd.query ( hashstring, resolve=feat)
        return df

def getConf(fam, index):
    if index == 50:
        print("Default")
        return "Default"
    with open("{}/runhistory.json".format(fam)) as json_data:
        data = json.load(json_data)
        return data["configs"]["{}".format(index + 1)]
    
def getInstanceCount(fam):
    df = pd.read_csv("ordered/{}.csv".format(fam))
    return len(df.keys()) - 2
def getScore(fam, index):

    with open("{}/runhistory.json".format(fam)) as json_data:
        data = json.load(json_data)
        return data["data"][index]
    
def getVirtual(group):
    df =pd.read_csv("ordered/{}.csv".format(group))
    performance = 0.0
    for i in range(1, len(df.keys())):
        key = df.keys()[i]
        #print(key)
        mindex = df[key].idxmin()
        #print(df["Config"][df[key].idxmin()])
        time = df[key][mindex]
        #print(time)
        performance += time
    return performance

def best2(fam):
    df = pd.read_csv("ordered/{}.csv".format(fam))

    minAss = (0,0)
    minDistr = (0,0)
    minimum = 3600 * 32
    for i in range(len(df["Score"])):
        
        for j in range(i + 1, len(df["Score"])):
            
            cost = 0.0
            k_i = 0
            k_j = 0
            for k in range(2, len(df.keys())):
                key = df.keys()[k]
                
                if float(df[key][i]) < float(df[key][j]):
                    cost += float(df[key][i])
                    k_i +=1
                else:
                    cost += float(df[key][j])
                    k_j += 1
            if cost < minimum:
                minimum = cost
                minAss = (i,j)
                minDistr = (k_i, k_j)
    return (minimum, minDistr)

def getBest(group):
    df =pd.read_csv("ordered/{}.csv".format(group))
    mindex = df["Score"].idxmin()
    return mindex

def getDefault(group):
    df1 = pd.read_csv("ordered/{}.csv".format(group))
    df = pd.read_csv("../defaults/liskov/results.csv")
    keys = df1.keys()[1:]
    s = 0
    for i in keys:
        s += df[i][0]
    return s

def getBestSet(group, keys):
    df =pd.read_csv("ordered/{}.csv".format(group))
    min = 898427589.0
    mindex = 50
    for i in range(len(df["Score"])):
        s = 0

        for key in keys:
            s += float(df[key][i])
        if s < min:
            min = s
            mindex = i
    return mindex, min

def evaluatemeta(metaindex, keys, a, minsize):
    vars = []
    feat = getFeat(keys)
    for key in keys:
        line = (feat[feat["hash"] == key])[feat.keys()[metaindex + 1]]
        vars.append((key, float(line)))
    vars.sort(key=lambda x: x[1])

    best = 476953454
    splitindex = 0
    configs = (0 , 0)
    lowerop = []
    upperop= []
    for i in range(minsize, len(vars) - minsize):
        if vars[i-1][1] == vars[i][1]:
            continue
        sum= 0
        lower = vars[:i]
        upper = vars[i:]
        conf1, cost1 = getBestSet(a, [x for x, _ in lower])
        conf2, cost2 = getBestSet(a, [x for x, _ in upper])
        sum += cost1 + cost2
        if sum < best:
            splitindex = i
            best = sum
            configs = (conf1, conf2)
            lowerop = lower
            upperop = upper
    return best, splitindex, configs, lowerop, upperop


def getBestScore(group):
    df =pd.read_csv("ordered/{}.csv".format(group))
    mindex = df["Score"].idxmin()
    s = 0
    for i in range(1, len(df.keys())):            
            s += df[df.keys()[i]][mindex]
    return df["Score"][mindex]

def beamsearch(fam, i, w, minstances):
    df = pd.read_csv("ordered/{}.csv".format(fam))
    n = len(df["Score"])
    beam = [(0, [])]  # (cost, [indices])
    
    for _ in range(i):
        new_beam = []
        for cost, indices in beam:
            for j in range(n):
                if j not in indices:
                    new_indices = indices + [j]
                    new_cost = 0.0
                    instance_counts = {idx: 0 for idx in new_indices}
                    for k in range(1, len(df.keys())):
                        key = df.keys()[k]
                        best_instance = min(new_indices, key=lambda idx: float(df[key][idx]))
                        instance_counts[best_instance] += 1
                        new_cost += float(df[key][best_instance])
                    
                    # Ensure each group has at least minstances instances
                    if all(count >= minstances for count in instance_counts.values()):
                        new_beam.append((new_cost, new_indices))
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:w]
    
    if not beam:
        return None, None, None, None
    
    best_cost, best_indices = beam[0]
    instance_counts = {idx: 0 for idx in best_indices}
    instances = {}
    for k in range(1, len(df.keys())):
        key = df.keys()[k]
        best_instance = min(best_indices, key=lambda idx: float(df[key][idx]))
        instance_counts[best_instance] += 1
        if not best_instance in instances.keys():
            instances[best_instance] = []
        instances[best_instance].append(key)
    
    return best_cost, best_indices, instance_counts, instances