import numpy as np


class PCA:

    def __init__(self,n_components=None):
      
        self.n_components = n_components  #number of components we want to keep
        self.components = None  #stores the directions
        self.meanof = None
        self.varience = None
        self.varience_ratio = None

    def fit(self,X,y):
        X = np.asarray(X,dtype=np.float32)

        #checks if the input is 2 dim
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array ,the shape is {X.shape}")
        
        number_of_samples , number_of_featues = X.shape
        
        max_comp = min(number_of_samples,number_of_featues)
