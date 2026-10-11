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

    def build_tree(self,X,y,depth):
        n_samples ,n_features = X.shape 
        n_labels = len(np.unique(y))

        if(n_labels == 1 or n_samples < self.min_samples or (self.max_depth is not None and depth >= self.max_depth)):
            leaf_value = self.freq_class(y)
            return Node(value=leaf_value)

        