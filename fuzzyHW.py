import pandas as pd
import numpy as np
from tqdm import tqdm

class fuzzyCmeans():
    """
    Supervised/Unsupervised C means

    if supervised:
        data label should be provided and K should be assigned.
        Grab K-NN for each sample to calculate their initial membership.
    else
        Random initial membership.

    Use C-centers to label data
    """

    def __init__(self, supervised=True, c=4, k=None, m=2, max_iter=10, threshold=None, print_log=True):
        self.sup = supervised
        self.c = c
        self.m = m
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold

        self.center = None
        self.print_log = print_log
        
        if supervised:
            if not isinstance(self.k, int):
                raise Exception('if supervisd, "k" should have type int')
            elif k <= 0:
                raise Exception('if supervisd, "k" should larger than 0')
            else:
                print("Supervised -- c value will be ignored. The number of clusters depends on input label.")
                print("Supervised -- calculate initial membership value based on K-NN")
        else:
            print("Unsupervised -- k calue will be ignored. Random initial membership value")

        if not isinstance(self.threshold, (int, float, type(None))):
            raise Exception('"threshold should be integer or float"')

        if not isinstance(self.c, int):
            raise Exception('"c" should have type int')

    # Fit data to the model        
    def fit(self, X, Y=None):
        """
        Compute k-means clustering.
        """
        # Check data type. X and Y must be data frame
        if type(X) != pd.core.frame.DataFrame or type(Y) != pd.core.series.Series:
            raise Exception('X should be pandas dataframe and Y should be 1D pandas series')
        elif self.sup:
            if  type(Y) != pd.core.series.Series:
                raise Exception('You should provide valid label if supervised.')
  
        self.X = X
        self.df = X.copy()
        self.dim = len(X.columns)
        self.n = X.shape[0]
        self.record_difference = []
        self.iteration = 0
      
        if self.sup:
            self.Y = Y
            self.df['Y'] = Y
            classes = list(set(Y))
            classes.sort()
            self.c = len(classes)
        else:
          classes = [i+1 for i in range(self.c)]
        self.classes = classes

        self.df['membership'] = self.__ini_membership()

        # Start iteration
        for iter in range(self.max_iter):
            if self.print_log:print("Iteration ",iter,end="-->")
            self.iteration += 1
            self.center = self.__compute_cluster_centers()
            update_membership, difference = self.__update_membership()
            if self.print_log:print(" Difference = ",difference)
            self.record_difference.append(difference)

            if self.threshold is not None:
                if difference < self.threshold:
                    self.df.membership = update_membership
                    break

            self.df.membership = update_membership
        
        clusterd_label = []
        for membership in self.df.membership:
            clusterd_label.append(membership.index(max(membership))+1)
        self.df['cluster'] = np.array(clusterd_label)

    def __compute_cluster_centers(self):
        """
        Compute new cluster centers

        Return:
            retva1: C centers
        """
        center = dict()
        for index,class_key in enumerate(self.classes):
            membership_list = np.array([mb[index] for mb in self.df.membership])
            membership_list = membership_list**self.m
            num = np.dot(membership_list, self.X)
            den = np.sum(membership_list)
            center[class_key] = num/den
        return center

    def __update_membership(self):
        """
        Update membership

        Return:
          retva1: updated membership value
        """
        memberships = []
        for index, sample in self.X.iterrows():
            num = []
            den = 0
            membership = []
            dist = [np.linalg.norm(sample-self.center[vj]) for vj in self.classes]
            for class_index in range(self.c):
                if dist[class_index] == 0:
                    num = [0 for i in range(self.c)]
                    num[class_index] = 1
                    den = 1
                    break
                else:
                    value = 1/dist[class_index]**(2/(self.m-1))
                    den += value
                    num.append(value)
            for class_index in range(self.c):
                membership.append(num[class_index]/den)
            memberships.append(membership)
        difference = self.__membership_diff(memberships)
        return memberships, difference

    def __membership_diff(self, memberships):
        origin = self.df.membership.to_numpy().tolist()
        difference = np.linalg.norm(np.array(memberships)-np.array(origin))
        return difference

    def __get_KNN(self, x, dataframe):
        """
        Get the K nearest neighbor W.R.T x and all labeled data

        Parameters:
            param1: one sample in labeled set
            param2: whole dataframe to search
        
        Return:
            retva1: dataframe that contains K nearest neighbor W.R.T. x
        """
        X = self.X.iloc[:,0:self.dim].to_numpy()
        dataframe['distance'] = [np.linalg.norm(Xi-x) for Xi in X]
        dataframe.sort_values(by='distance', ascending=True, inplace=True)
        return dataframe.iloc[0:self.k]
        
    def __ini_membership(self):
        """
        Generate Initial Membership Value

        Description:
            if supervised, calculate membership value based on surrounding K nearest neighbor and their label
            if unsupervised, random
        """
        memberships = []
        if self.print_log:print("Initial membership")
        pbar = tqdm(total=self.n)
        if self.sup:
            # calculate μ-ij, the membership of jth sample w.r.t. ith class
            for j in range(self.n):
                pbar.update(1)
                
                x = self.X.iloc[j]
                y = self.Y.iloc[j]
                neighbor = self.__get_KNN(x.to_numpy(), self.df.copy())
                count = neighbor['Y'].value_counts().to_dict()

                sample_membership = []
                for i in self.classes:
                    try:
                        uij = (count[i] / self.k) * 0.49
                        if i == y:
                            uij = uij + 0.51
                        sample_membership.append(uij)
                    except Exception as e:
                        sample_membership.append(0)
                memberships.append(sample_membership)
        else:
          for i in range(self.n):
              pbar.update(1)

              rand = np.random.dirichlet(np.ones(self.c),size=1)[0]
              memberships.append(rand.tolist())
        
        # Close progressing bar      
        pbar.close()

        return memberships

class fuzzyKNN():
    """
    User should fit the clustered center to the classifier.
    Unlabeled data will be predicted W.R.T. the centers obtained from FCM
    """
    def __init__(self, k=4, m=2, a=0.51):
        self.k = k
        self.m = m
        self.a = a

        if not isinstance(self.k,int):
            raise Exception('"k" should have type int')
        elif not isinstance(self.m,(int, float)):
            raise Exception('"m" should have type int')
        elif not isinstance(self.a,(int, float)):
            raise Exception('"a" should have type int')
    
    # Fit data to the model        
    def fit(self, center):
        """
        Fit center to classifier
        
        Parameters
            param1: centers in dictionary type. {class name:center}
        """
        # Check data type. X and Y must be data frame
        if type(center) != dict:
            raise Exception('X should be dictionary, the value should be class center')

        self.center = list(center.values())
        self.keys = list(center.keys())
        self.classes = len(center)
        self.classmap = {i:self.keys[i] for i in range(self.classes)}    

        if self.classes < self.k:
            raise Exception("K should not larger than class number")

        self.membership = self.__cal_membership()
       
    # Predict
    def predict(self, new_x):
        """
        Predict unlabeled data

        Parameters
            param1: new data in pandas dataframe form

        Return
            retva1: Predict result list
        """
        if type(new_x) != pd.core.frame.DataFrame:
            raise Exception('X should be pandas dataframe.')

        pred = []
        for index, x in new_x.iterrows():
            # Get k neighbor
            k_prototype_index, distance = self.__get_KNN(x.to_numpy())

            # Calculate the membership of x in i-th class
            memberships = []
            for i in range(self.classes):           
                den = 0
                num = 0

                # Iterate through k prototypes
                for j in k_prototype_index:

                    # The membership of j-th prototype W.R.T. i-th prototype
                    uji = self.membership[j][i]
                    dist = distance[j]
                    if dist != 0:
                        den = den + 1 / dist**(2/(self.m-1))
                        num = num + uji * 1 / dist**(2/(self.m-1))
                    else:
                        num = uji
                        den = 1
                        break

                membership = num/den
                memberships.append(membership)
            max_class = memberships.index(max(memberships))
            pred.append(self.classmap[max_class])
        return pred

    # Get kth nearest neighbor
    def __get_KNN(self, x):
        """
        (Private) Get K nearnest neighbor

        Parameters
            param1: Point of interest

        Return
            retva1: Sorted index and value
        """
        center = np.array(self.center)
        distance = [np.linalg.norm(prototype-x) for prototype in center]
        sort_index = np.argsort(distance)
        return sort_index[0:self.k], distance
        
    # Calculate membership between each class prototype
    def __cal_membership(self):
        """
        (Private) Calculate membership of each prototype

        Return
            retva1:memberships list of each prototype
        """
        memberships = []
        # calculate μ-ij, the membership of jth class prototype w.r.t ith class prototype
        for j in range(self.classes):
            dist = [np.linalg.norm(prototype-self.center[j]) for prototype in self.center]
            den = sum([1/distance**(2/(self.m-1)) for distance in dist if distance != 0])

            prototype_membership = []
            for i in range(self.classes):
                if i == j:
                    prototype_membership.append(self.a)
                else:
                    num = 1/dist[i]**(2/(self.m-1))
                    prototype_membership.append(num/den*(1-self.a))
            memberships.append(prototype_membership)
        return memberships

def clust_to_class(label_x, label_y, cluster_y):
    """
    Determine the mapping from class classified by C Means to label

    Parameters:
      param1: known data
      param2: known class
      param3: cluster class
    """
    labeled = label_x.copy()
    labeled['sol'] = label_y
    labeled['cluster'] = cluster_y

    labeled_classes = list(set(label_y))
    labeled_classes.sort()
    cluster_classes = list(set(cluster_y))
    cluster_classes.sort()

    relation = dict()
    for c_clust in cluster_classes:
        ratio_list = []
        for l_class in labeled_classes:
            l_class_df = labeled.loc[labeled['sol'] == l_class]         
            c_in_l = l_class_df.loc[l_class_df['cluster'] == c_clust] # The l class element in clust c cluster
            c_clust_df = labeled.loc[labeled['cluster'] == c_clust] # The data that is clusted as c_clust
            ratio = c_in_l.shape[0]/c_clust_df.shape[0]
            ratio_list.append(ratio)
        max_ratio = max(ratio_list)
        clust_to_class = ratio_list.index(max_ratio)
        relation[c_clust] = labeled_classes[clust_to_class]
        print("Class ", labeled_classes[clust_to_class]," in Cluster ",c_clust," -> ",max_ratio*100,"%")
    return relation