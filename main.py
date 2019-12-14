from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

from fuzzyHW import *

# Fuzzy C Means parameter
FCM_SUPPERVISED = False
FCM_C = 4
FCM_K = 10 # unused if unsupervised
FIC_M = 2
FCM_ITER = 30
FCM_THRESHOLD = 0.1

# Fuzzy KNN parameter
FKN_K = 4
FKN_M = 2
FKN_A = 0.6


def get_pca(train_x, test_x, component=2):
    pca = PCA(n_components=component)
    pca.fit(train_x)
    train_pca = pca.transform(train_x)
    test_pca = pca.transform(test_x)
    return train_pca, test_pca

def get_lda(train_x, train_y, test_x, component=2):
    clf = LinearDiscriminantAnalysis(n_components=component)
    clf.fit(train_x,train_y)
    train_lda = clf.transform(train_x)
    test_lda = clf.transform(test_x)
    return train_lda, test_lda

def get_rmvar(train_x, test_x, threshold=20):
    selector = VarianceThreshold(threshold=20)
    selector.fit(train_x)
    train_var = selector.transform(train_x)
    test_var = selector.transform(test_x)
    return train_var, test_var

if __name__ == "__main__":

    # Read Data
    data = pd.read_csv('wifi_localization.txt', sep="\t", header=None, names=['A', 'B','C','D','E','F','G','sol'])
    print(data)

    # Split data -> In this case, training data as testing data
    train_x = data[['A', 'B','C','D','E','F','G']]
    train_y = data['sol']
    test_x = data[['A', 'B','C','D','E','F','G']]
    test_y = data['sol']


    # Uncomment below to select specific dimension reduction method
    train_x, test_x = train_x, test_x
    #train_x, test_x = get_pca(train_x, test_x, component=2)
    #train_x, test_x = get_lda(train_x, train_y, test_x, component=2)
    #train_x, test_x = get_rmvar(train_x, test_x, threshold=20)

    # Fuzzy C-Means cluster
    cluster = fuzzyCmeans(supervised=FCM_SUPPERVISED, c=FCM_C, k=FCM_K, m=FIC_M, max_iter=FCM_ITER, threshold=FCM_THRESHOLD)
    cluster.fit(train_x, train_y)

    # Get the map from cluster class to real class
    relation = clust_to_class(data.iloc[:,0:7], data['sol'], cluster.df['cluster'])
    
    # Cluster scoring
    map_df = cluster.df.copy()
    map_df['cluster'] = np.array([relation[clust] for clust in map_df['cluster']])
    accuracy = np.mean(map_df.cluster==train_y)
    print('Accuracy',accuracy)
    print(cluster.center)

    # Fuzzy KNN

    # Get cluster center and fit to fuzzy KNN
    center = cluster.center
    classifier = fuzzyKNN(k=FKN_K, m=FKN_M, a=FKN_A)
    classifier.fit(center)

    # Predict
    sol= classifier.predict(test_x)

    # Mapping
    sol = [relation[s] for s in sol]

    # Scoring
    classifier_accuracy = np.mean(sol==test_y)
    print("Predict result")
    print(sol)
    print('Classifier Accuracy',classifier_accuracy)