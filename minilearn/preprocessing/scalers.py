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


class MinMaxScaler:
    def __init__(self,dtype=np.float32):
        self.eps = np.finfo(dtype).eps
        self.min = None
        self.max =None

    # set the values of X min and X max
    def fit(self,X):
        self.min = np.min(X,axis=0)
        self.max = np.max(X,axis=0)
        return self

    # (x - xmin )/ (xmax - xmin + eps)
    def transforme(self,X):
        return (X - self.min)/ (self.max - self.min + self.eps)

    def fit_transform(self,X):
        return self.fit(X).transforme(X)


class  MaxAbsScaler:
    def __init__(self,dtype=np.float32):
        self.eps = np.finfo(dtype).eps
        self.max_abs = None
    #set max(|x|)
    def fit(self,X):
        self.max_abs = np.max(np.abs(X),axis=0)
        return self

    # x / max(|x|) + eps
    def transform(self,X):
        return X /(self.max_abs + self.eps)

    
    def fit_transform(self,X):
        return self.fit(X).transform(X)
    
    
