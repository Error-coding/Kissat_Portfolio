
with open("defaults.txt") as defaults:
    with open("runtimes.txt") as runtimes:
        for i in range(35):
            runtime = runtimes.readline()[:-1]
            default = defaults.readline()[:-1]
            print("{} {} {}".format(runtime, default, float(runtime) / float(default)))