import csv
import numpy as np
import sys

group = int(sys.argv[1])

timeout = 2 * 1000

fields = ["Config", "Score", "Mean"]

data = []
times = []
sample = {}

with open("./raw/{}.out".format(group)) as raw:
    for line in raw:
        if "Config" in line:
            sample["Config"] = line.split(" ")[1][2:-1]


        if "seconds" in line:
            inst = line.split(" ")[2].split("/")[-1].split("-")[0]

            if not inst in fields:
                fields.append(inst)

            time = float(line.split(" ")[-2])

            if time == 10000:
                time = timeout

            times.append(time)
            sample[inst] = time

        if "score" in line:
            sample["Score"] = float(line.split(" ")[-1])
            sample["Mean"] = np.mean(times)

            data.append(sample)
            times = []
            sample = {}


with open("./ordered/{}.csv".format(group), mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)

    writer.writeheader()
    writer.writerows(data)