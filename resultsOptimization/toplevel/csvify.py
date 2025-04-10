import csv
import numpy as np
import sys
import pandas as pd

group = int(sys.argv[1])

timeout = 2 * 1800

fields = ["Score"]

data = []
times = []
sample = {}
s = 0
with open("./{}/log.out".format(group)) as raw:
    for line in raw:

        if "seconds" in line:
            inst = line.split(" ")[2].split("/")[-1].split("-")[0]

            if not inst in fields:
                fields.append(inst)

            time = float(line.split(" ")[-2])

            if time == 10000:
                time = timeout

            s += time
            times.append(time)
            sample[inst] = time

        if "Starting pool" in line or "Finished" in line:
            if len(times) != 0:
                sample["Score"] = s

                data.append(sample)
                times = []
                sample = {}
                s= 0

df = pd.read_csv("../defaults/liskov/results.csv")

for i in fields:
    if i == "Score":
        continue
    sample[i] = df[i][0]
    s += float(df[i][0])
sample["Score"] = s
data.append(sample)


with open("./ordered/{}.csv".format(group), mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)

    writer.writeheader()
    writer.writerows(data)