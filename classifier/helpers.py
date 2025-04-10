import json

from sklearn.cluster import KMeans
from gbd_core.api import GBD

import warnings

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from itertools import combinations



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

# Returns virtual performance for a set of instances and a set of configurations
def getVirtual(insts, confs):
    df = pd.read_csv("top40.csv")
    filtered_df = df[df['key'].isin(insts) & df['configuration'].isin(confs)]
    min_runtimes = filtered_df.groupby('key')['time'].min()
    min_runtimes = min_runtimes.where(min_runtimes < 1800, 3600)
    return min_runtimes.sum() / len(insts)

# Returns the performance of the Default configuration on a set of instances
def getDefault(insts):
    df = pd.read_csv("top40.csv")
    filtered_df = df[df['key'].isin(insts) & (df['configuration'] == 'Default')]
    filtered_df.loc[filtered_df['time'] >= 1800, 'time'] = 3600
    
    return filtered_df['time'].sum() / len(insts)


# Given a set of instances paired with a predicted index and given a list of configurations pointed to by the index, returns the PAR-2 score of the prediction
def eval(hashpreds, confs):
    df = pd.read_csv("top40.csv")
    hashes = hashpreds["hash"].tolist()
    preds = hashpreds["predicted_index"].tolist()
    #print(hashes)
    sum = 0
    for i in range(len(hashes)):
        conf = confs[preds[i]]
        entr = df[(df['key'] == hashes[i]) & (df['configuration'] == conf)]
        if not entr.empty:
            if float(entr['time'].values[0]) <= 1800:
                sum += float(entr['time'].values[0])
            else:
                sum += 3600
    return sum / len(hashpreds)

# Given a set of instances paired with a predicted configuration, returns the PAR-2 score of the prediction
def evalConfigPredict(hashpreds):
    df = pd.read_csv("top40.csv")
    hashes = hashpreds["hash"].tolist()
    preds = hashpreds["predicted_index"].tolist()
    #print(hashes)
    sum = 0
    for i in range(len(hashes)):
        conf = preds[i]
        entr = df[(df['key'] == hashes[i]) & (df['configuration'] == conf)]
        if not entr.empty:
            if float(entr['time'].values[0]) <= 1800:
                sum += float(entr['time'].values[0])
            else:
                sum += 3600
        else:
            print("Something went wrong")
    return sum / len(hashpreds)

# Returns the features used in classification.
def get_available_features():
    with GBD([ "/home/raphael-zipperer/Uni/BA/database/base.db" ]) as gbd:
        feat = gbd.get_features('base')
        # Filtered out because not objective oblivious
        feat.remove("base_features_runtime")
        # Filtered out because incomplete, not available for every instance
        feat.remove("ccs")
        feat.remove("bytes")
        return feat


def get_dataset_by_hashes(hashes_list):
    with GBD(['/home/raphael-zipperer/Uni/BA/database/base.db', '/home/raphael-zipperer/Uni/BA/database/meta.db']) as gbd:
        features = get_available_features()
        all_hashes = [hash for sublist in hashes_list for hash in sublist]
        df = gbd.query('(track=main_2023 or track=main_2024) and minisat1m!=yes', resolve = features)
        df = df[df['hash'].isin(all_hashes)]
        df[features] = df[features].apply(pd.to_numeric)
        df['index'] = -1
        for i, sublist in enumerate(hashes_list):
            df.loc[df['hash'].isin(sublist), 'index'] = i
        return df

def get_dataset_by_fams(famlist):
    with GBD(['/home/raphael-zipperer/Uni/BA/database/base.db', '/home/raphael-zipperer/Uni/BA/database/meta.db']) as gbd:
        features = get_available_features()
        df = gbd.query('(track=main_2023 or track=main_2024) and minisat1m!=yes', resolve = features + ["family"])
        

        df[features] = df[features].apply(pd.to_numeric)
        df['index'] = -1
        for i, sublist in enumerate(famlist):
            df.loc[df['family'].isin(sublist), 'index'] = i

        

        return df

def get_prediction_dataset(features, target):
    with GBD([ '/home/raphael-zipperer/Uni/BA/database/base.db', '/home/raphael-zipperer/Uni/BA/database/meta.db' ]) as gbd:
        df = gbd.query('(track=main_2023 or track=main_2024) and minisat1m!=yes', resolve=features+[target])
        df[features] = df[features].apply(pd.to_numeric)

        return df


# Returns all configurations we work with in training
def getAllConfigs():
    df = pd.read_csv("top40.csv")
    return df['configuration'].unique().tolist()

# Returns the SBS over a set of instances and an optional set of configurations
def getBest(insts, configs=getAllConfigs()):
    df = pd.read_csv("top40.csv")
    df = df[df['key'].isin(insts)]
    df = df[df['configuration'].isin(configs)]
    df.loc[df['time'] >= 1800, 'time'] = 3600
    df = df.groupby('configuration')['time'].sum()

    best_config = df.idxmin()

    return best_config

# expects a relabeled dataframe with features
def evalClasses(data, configs, numsamples):
    feat = get_available_features()

    results = []
    for i in range(numsamples):
        X_train, X_test, y_train, y_test = train_test_split(data[feat + ["hash"]], data['index'], test_size=0.2, random_state=i, stratify=data['index'])


        model = RandomForestClassifier()
        model.fit(X_train[feat], y_train)
        y_pred = model.predict(X_test[feat])
        accuracy = accuracy_score(y_test, y_pred)

        insts = X_test['hash'].tolist()
        predictions_df = pd.DataFrame(list(zip(X_test["hash"], y_pred)), columns=["hash", "predicted_index"])

        default_sum = getDefault(insts)
        virtual_sum = getVirtual(insts, configs)
        eval_sum = eval(predictions_df, configs)

        results.append({
            'Default': default_sum,
            'Virtual': virtual_sum,
            'Eval': eval_sum
        })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    results_df.boxplot(column=['Default', 'Virtual', 'Eval'], showmeans=True, meanline=True)
    plt.title('Boxplot of Evaluation Metrics')
    plt.ylabel('Values')
    plt.xlabel('Metrics')
    plt.grid(True)

    plt.show()
    print(results_df.mean())

    return results


def genericClusterEval(data, algorithm, numsamples, indexed=False):
    features = get_available_features()
    
    results = []

    for i in range(numsamples):
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=i, stratify=data['family'])
        
        #if indexed:
        #    classifier = algorithm(data_train[features + ["hash", "index"]])
        #else:
        kmeans, cluster_config = algorithm(data_train[features + ["hash"]])
        
        #data_test['cluster'] = kmeans.predict(data_test[features])


        # Get the labels of the closest points
        data_test['cluster'] = kmeans.predict(data_test[features])

        insts = data_test['hash'].tolist()
        y_pred = [cluster_config[cluster] for cluster in data_test['cluster']]

        hash_prediction = pd.DataFrame(list(zip(data_test["hash"], y_pred)), columns=["hash", "predicted_index"])

        #print(hash_prediction)

        distinct_configurations = list(set(cluster_config.values()))
        default = getDefault(data_test["hash"].tolist())
        virtual = getVirtual(data_test["hash"].tolist(), distinct_configurations)
        eval = evalConfigPredict(hash_prediction)

        results.append({
            'Default': default,
            'Virtual': virtual,
            'Eval': eval
        })
    return results


# takes as input a list of experiments, with each experiment containing a list of data measured by using different seeds, each containing default, virtual and solver performance respectively
def compareModels(experiments, names=[i for i in range(50)], title="Boxplot of Evaluation Data", default=False, defvbs=False):
    #virtual_data = [[entry['Virtual'] for entry in result] for result in experiments]
    #default_data = [[entry['Default'] for entry in result] for result in experiments]
    eval_data = [[entry['Eval'] for entry in result] for result in experiments]
    virtual_data = [[entry['Virtual'] for entry in result] for result in experiments]
    default_data = [[entry['Default'] for entry in result] for result in experiments]


    plt.figure(figsize=(10, 6))
    plt.boxplot(eval_data, patch_artist=True,showmeans=True, meanline=True)
    if defvbs:
        plt.boxplot(default_data[0], patch_artist=True, showmeans=True, meanline=True, positions=[len(eval_data) + 1],widths=0.5)
        plt.boxplot(virtual_data[0], patch_artist=True, showmeans=True, meanline=True, positions=[len(eval_data) + 2],widths=0.5)
        names.append("Default")
        names.append("Virtual")
    if default:
        plt.boxplot(default_data[0], patch_artist=True, showmeans=True, meanline=True, positions=[len(eval_data) + 1],widths=0.5)
        names.append("Default")
    plt.legend()
    #plt.title(title)
    plt.ylabel('PAR-2 score')
    #plt.xlabel('Portfolio size')
    plt.xticks(ticks=range(1, len(names) + 1), labels=names, rotation=90)
    plt.show()

    return [sum(result)/float(len(result)) for result in eval_data]

def virtualDefaultEval(experiment):
    eval_data = [entry['Eval'] for entry in experiment]
    virtual_data = [entry['Virtual'] for entry in experiment]
    default_data = [entry['Default'] for entry in experiment]

    plt.figure(figsize=(10, 6))
    plt.boxplot([eval_data, virtual_data, default_data], patch_artist=True, showmeans=True, meanline=True, labels=['Classifier', 'VBS', 'Default'])
    #plt.title(f'Comparison of {name}')
    plt.ylabel('PAR-2 score')
    plt.grid(True)
    plt.show()

def compareVirtual(experiments, names=[i for i in range(50)], nrows=1, ncols=1):
    virtual_data = [[entry['Virtual'] for entry in result] for result in experiments]


    plt.figure(figsize=(10, 6))
    plt.boxplot(virtual_data, showmeans=True, meanline=True)
    plt.title('Boxplot of Evaluation Data')
    plt.ylabel('Values')
    plt.xlabel('Experiments')
    plt.xticks(ticks=range(1, len(names) + 1), labels=names, rotation=90)
    plt.grid(True)
    plt.show()


def configScores(instances):
    df = pd.read_csv('../classifier/top40.csv')

    df = df[df['key'].isin(instances)]
    df.loc[df['time'] > 1800, 'time'] = 3600
    config_time_sum = df.groupby('configuration')['time'].sum()
    return config_time_sum

def cluster(instances, nclusters=7):
    config_time_sum = configScores(instances)
    configurations = config_time_sum.index.values.reshape(-1, 1)
    runtimes = config_time_sum.values.reshape(-1, 1)

    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(runtimes)
    labels = kmeans.labels_

    clustered_configs = pd.DataFrame({'Configuration': configurations.flatten(), 'Runtime': runtimes.flatten(), 'Cluster': labels})
    return clustered_configs

def bestcluster(instances, nclusters=7, seed=0):
    clustered_configs = cluster(instances, nclusters)
    non_empty_clusters = clustered_configs.groupby('Cluster').filter(lambda x: len(x) > 0)
    min_runtime_cluster = non_empty_clusters.groupby('Cluster')['Runtime'].mean().idxmin()
    return non_empty_clusters[non_empty_clusters['Cluster'] == min_runtime_cluster]

def minhitset(data_train, k, seed=0, target="family"):
    warnings.filterwarnings('ignore')

    data = data_train.copy()
    hashes_by_family = data.groupby(target)["hash"].apply(list).to_dict()
    
    famindeces = {}
    for fam, hashes in hashes_by_family.items():
        indeces = bestcluster(hashes, nclusters=k, seed=seed).index.tolist()
        famindeces[fam] = indeces.copy()
    
    # extracts indeces that must be contained in any hitting set (meaning that there is a set with only one element that contains the entry)
    unique_indices = []
    for fam, indeces in famindeces.items():
        if len(indeces) == 1 and indeces[0] not in unique_indices:
            unique_indices.append(indeces[0])

    unique_indices.sort()

    
    given = unique_indices


    def is_hitting_set(sets, candidate):
        return all(any(elem in candidate for elem in s) for s in sets)

    def minimum_hitting_set(sets, given):
        all_elements = set(elem for s in sets for elem in s if elem not in given)
        for size in range(0, len(all_elements) + 1):

            for candidate in combinations(all_elements, size):
                candidate_set = set(candidate).union(given)
                if is_hitting_set(sets, candidate_set):
                    return candidate_set
        return all_elements.union(given)

    filtered_sets = [s for s in famindeces.values() if not any(elem in given for elem in s)]
    hitting_set = minimum_hitting_set(filtered_sets, given)
    return list(hitting_set)

def locConfig(c):
    df = pd.read_csv('./top40.csv')
    df = df.groupby('configuration')['time'].sum()

    configs = df.keys().tolist()
    index = configs.index(c)
    return index