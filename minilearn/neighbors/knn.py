import numpy as np

class  KNClassifier:
    def __init__(self,n_neighbors = 10):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self,X,y):
        X = np.asarray(X,dtype=np.float32)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self
    
    def pdist(self,X):
        dot_product = X @ self.X_train.T
        x_2 = np.sum(X ** 2 ,axis = 1,keepdims=True)
        train_2 = np.sum(self.X_train ** 2 ,axis=1,keepdims=True).T

        dist = x_2 + train_2 - 2.0 * dot_product

        return np.maximum(dist ,0.0)

    def predict(self,X):
        X = np.asarray(X,dtype=np.float32)
        dist = self.pdist(X)
        k = min(self.n_neighbors ,len(self.X_train))
        neighbord_idx = np.argpartition(dist,kth = k - 1,axis = 1)[: , :k]
        neighbor_labels = self.y_train[neighbord_idx]

        preds = []
        for row in neighbor_labels:
            val,counts = np.unique(row,return_counts=True)
            preds.append(val[np.argmax(counts)])

        return np.array(preds,dtype=self.y_train.dtype)

    def score(self,X,y):
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))



