import json


for i in range(35):
    with open('{}/runhistory.json'.format(i)) as js:
        obj= json.load(js)
        min = 100000 * 20
        for k in obj["data"]:
            if float(k[4]) < min:
                min = k[4]
        print(min)