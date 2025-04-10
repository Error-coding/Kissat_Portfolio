import csv
import numpy as np
import sys
import os

#def addAll(f):



dir = "./" + str(sys.argv[1])

timeout = 2 * 1800.0


data = {}

out_files = [f for f in os.listdir(dir) if f.endswith('.out')]

for group in out_files:
    with open(dir + "/" + group) as raw:
        for line in raw:
            if line.startswith('['):
                parts = line.strip('[]\n').split(',')
                hashes = list(map(lambda x: x.split("/")[-1].split("-")[0], parts))
                for h in hashes:
                    if h not in data.keys():
                        data[h] = timeout

            if "seconds" in line and "Terminated" not in line:
                inst = line.split(" ")[1].split("/")[-1].split("-")[0]

                time = float(line.split(" ")[-2])

                data[inst] = time

csv_file = dir + "/results.csv"

with open(csv_file, mode='w', newline='') as f:
    w = csv.DictWriter(f, data.keys())
    w.writeheader()
    w.writerow(data)



