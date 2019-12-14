# Fuzzy-C-means-KNN

This is the assignment of Advanced Fuzzy Control.  
The repo provide a Python implementation of Fuzzy C-means (FCM) and Fuzzy-KNN (FKN).  

The FKN in this repo is not a normal FKN. One should cluster the labeled data and get the class prototype through FCM and then fit the class prototype to FKN. The FKN algorithm will predict the unlabeled data W.R.T. the class prototype. Algorithm choose K Nearnest prototype of the unlabeled data and calculate their relationship.  

Flow:

    1. Fit data into FCM. If supervised, remember to feed train_y into cluster.
    2. After 1., onee can obtain cluster result through cluster.df
    3. Execute  clust_to_class function to obtain the relationship between cluster class and real class.
    4. Fit cluster.center to FKN.
    5. Feed unlabeled data(in pandas dataframe type) to FKN to predict which cluster it belongs to.
    6. Map the FKN result to real class through the relationship you just obtained.
