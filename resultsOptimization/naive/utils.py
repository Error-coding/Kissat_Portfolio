import json
import pandas as pd
import matplotlib.pyplot as plt



def getBest(group):
    df =pd.read_csv("ordered/{}.csv".format(group))
    mindex = df["Score"].idxmin()
    return mindex


def plotcummulativeV(group):
    df =pd.read_csv("ordered/{}.csv".format(group))
    timesV = []
    for i in range(3, len(df.keys())):
        key = df.keys()[i]
        mindex = df[key].idxmin()
        time = df[key][mindex]
        timesV.append(float(time))

    timesBest = []
    best = getBest(group)
    for i in range(3, len(df.keys())):
        timesBest.append(float(df[df.keys()[i]][best]))

    minimum = 2000 * 16
    timesApprox = []
    for i in range(len(df["Config"])):
        
        for j in range(i + 1, len(df["Config"])):
            timesApproxTemp = []
            cost = 0.0
            for k in range(3, len(df.keys())):
                key = df.keys()[k]
                
                if float(df[key][i]) < float(df[key][j]):
                    cost += float(df[key][i])
                    timesApproxTemp.append(float(df[key][i]))
                else:
                    cost += float(df[key][j])
                    timesApproxTemp.append(float(df[key][j]))
            if cost < minimum:
                minimum = cost
                timesApprox = timesApproxTemp
    timesApprox.sort()




    cum = list(range(1, len(df.keys()) - 2))
    timesV.sort()
    timesBest.sort()

    plt.step(timesV, cum, where='post', label='Virutal Solver', linestyle='-', color='blue')
    plt.step(timesBest, cum, where='post', label='Best Average Configuration', linestyle='-', color='red')
    plt.step(timesApprox, cum, where='post', label='V Solver Approximation', linestyle='-', color='green')

    # Achsen beschriften
    plt.xlabel('Zeit (Sekunden)')
    plt.ylabel('# solved')
    plt.title('Kumulativer Graph der Ereignisse')

    # Graph anzeigen
    plt.legend()
    plt.show()

def best2(fam):
    df = pd.read_csv("ordered/{}.csv".format(fam))

    minAss = (0,0)
    minDistr = (0,0)
    minimum = 2000 * 16
    for i in range(len(df["Config"])):
        
        for j in range(i + 1, len(df["Config"])):
            
            cost = 0.0
            k_i = 0
            k_j = 0
            for k in range(3, len(df.keys())):
                key = df.keys()[k]
                
                if float(df[key][i]) < float(df[key][j]):
                    cost += float(df[key][i])
                    k_i +=1
                else:
                    cost += float(df[key][j])
                    k_j += 1
            if cost < minimum:
                minimum = cost
                minAss = (df["Config"][i],df["Config"][j])
                minDistr = (k_i, k_j)
    #print(df["Config"][minAss[0]])
    #print(df["Config"][minAss[1]])
    #print("{}: Min {}, Distribution {}".format(fam, minimum, minDistr))
    #print(minimum)
    #print(minDistr)
    return (minimum, minDistr, minAss)

def getS(group):
    df = pd.read_csv("ordered/{}.csv".format(group))
    mindex = df["Score"].idxmin()
    s = 0
    for i in range(3, len(df.keys())):
        key = df.keys()[i]
        s += float(df[key][mindex])
    return s