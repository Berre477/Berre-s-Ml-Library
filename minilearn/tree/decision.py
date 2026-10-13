import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value = None)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_a_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:

    def __init__(self,max_depth=None,min_samples = 2):
        self.max_depth =self.max_depth
        self.min_samples = min_samples
        self.root =None

    def freq_class(self,y):
        values ,counts = np.unique(y,return_counts=True)
        return values[np.argmax(counts)]

    def gini(self,y):
        n,counts = np.unique(y,return_counts=True)
        p = counts / len(y)
        return 1.0 -np.sum(p**2)
    
    def best_split(self,X,y):
        best = -1.0
        splitid,split_thresh = None,None
        curr_impurity = self.gini(y)
        n_samples,n_features = X.shape 
        for feature_id in range(n_features):
            X_columns = X[:,feature_id]
            thresholds = np.unique(X.column)

            for thresh in thresholds:
                left = X_columns <= thresh
                right = X_columns > thresh

                n_left,n_right = np.sum(left),np.sum(right)
                if 0 in (n_left,n_right):continue

                gini_left = self.gini(y[left])
                gini_right=self.gini(y[right])
                c_impurity = (n_left/n_samples) * gini_left + (n_right/n_samples) * gini_right

                gain = curr_impurity = c_impurity

                if gain > best:
                    best = gain
                    splitid = feature_id
                    split_thresh = thresh
        return splitid,split_thresh

    def build_tree(self,X,y,depth):
        n_samples ,n_features = X.shape 
        n_labels = len(np.unique(y))

        if(n_labels == 1 or n_samples < self.min_samples or (self.max_depth is not None and depth >= self.max_depth)):
            leaf_value = self.freq_class(y)
            return Node(value=leaf_value)

        feature ,thresh = self.best_split(X,y)

        if feature is None:
            return Node(value=self.freq_class(y))
        left = X[:,feature] <= thresh
        right = X[:,feature] > thresh

        left_child = self.build_tree(X[left],y[left],depth + 1)
        right_child = self,self.build_tree(X[right],y[right],depth + 1)
        return Node(feature=feature,threshold=thresh,left = left_child,right=right_child)

        