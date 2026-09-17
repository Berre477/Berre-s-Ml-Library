import numpy as np 

class StandartScaler:
    def __init__(self,dtype=np.float32):
        self.eps = np.finfo(dtype).eps
        self.mean = None
        self.std = None

    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis = 0)
        return self

    def transform(self,X):
        return (X -self.mean) / (self.std + self.eps)