# Fuzzy-C-means-KNN

This is the assignment of Advanced Fuzzy Control.  
The repo provide a Python implementation of Fuzzy C-means (FCM) and Fuzzy-KNN (FKN).  

### The data we use to evaluate the algorithm is the Wireless Indoor Localization Data Set from [UCI](https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization)

## Introduction

The FKN in this repo is not a normal FKN. One should cluster the labeled data and get the class prototype through FCM and then fit the class prototype to FKN. The FKN algorithm will predict the unlabeled data W.R.T. the class prototype. Algorithm choose K Nearnest prototype of the unlabeled data and calculate their relationship.  

Flow:

    1. Fit data into FCM. If supervised, remember to feed train_y into cluster.
    2. After 1., onee can obtain cluster result through cluster.df
    3. Execute  clust_to_class function to obtain the relationship between cluster class and real class.
    4. Fit cluster.center to FKN.
    5. Feed unlabeled data(in pandas dataframe type) to FKN to predict which cluster it belongs to.
    6. Map the FKN result to real class through the relationship you just obtained.

## Another Approach

An another implementation of FKN (A-FKN) is also provided. This FKN doesn't need the cluster. One should fit labeled data and label into A-FKN. The algorithm will serve all data as labeled center and calculate their membership value. When a new data is feed, algorithm will find the K Nearnest labeled data W.R.T. new data and calculate the membership between new data and selected labeled data. The detail is in [1]

Flow:

    1. Prepare data (x in pandas dataframe format and y in pandas series format)
    2. Fit x and y into FKN
    3. Predct new data

The example and implementation are both in AnotherFKN.

## Reference

[1]	J. M. Keller, M. R. Gary, and J. A. Giviens, Jr. “A fuzzy K -nearest neighbor algorithm”, IEEE Trans. Systems, Men, & Cybernetics, SMC-15, No. 4, pp. 580-585, 1985
