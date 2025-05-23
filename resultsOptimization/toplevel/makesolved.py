import sys
import csv


timeout = 2 * 1800

dic = {0:"TO", 10:"SAT", 20:"UNSAT"}
fields = []
values = {}

for group in range(35):
    with open("./{}/log.out".format(group)) as raw:
        code = 0
        for line in raw:
            if "expected code" in line:
                code = int(line.split(",")[0][-2:])


            if "seconds" in line:
                inst = line.split(" ")[2].split("/")[-1].split("-")[0]

                if not inst in fields:
                    fields.append(inst)
                    values[inst] = dic[code]

                if code != 0:
                    values[inst] = dic[code]
                code = 0

    with open("./status/{}.csv".format(group), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()
        writer.writerows([values])   