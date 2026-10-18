import numpy as np


class PCA:

    def __init__(self,n_components=None):
      
        self.n_components = n_components  #number of components we want to keep
        self.components = None  #stores the directions
        self.mean_features = None
        self.varience = None
        self.varience_ratio = None

    def fit(self,X,y):
        X = np.asarray(X,dtype=np.float32)

        #checks if the input is 2 dim
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array ,the shape is {X.shape}")
        
        number_of_samples , number_of_featues = X.shape
        max_comp = min(number_of_samples,number_of_featues)

        if self.n_components is None: number_of_components = max_comp
        else: number_of_components = min(int(self.n_components),max_comp)

        self.mean_features = np.mean(X,axis=0)

        centered_data = X - self.mean_features

        #Applying SVD to find the directions
        _,singular_val , directions = np.linalg.svd(centered_data,full_matrices=False)

        #finds the largest abs val in each component
        largest_val_idx = np.argmax(np.abs(directions),axis = 1)
        row_idx = np.arange(directions.shape[0])
        correct = np.sign(directions[row_idx,largest_val_idx])

        self.components = directions[:number_of_components]

        #gets the eigenvalues of the covariance od the matrx singular_val **2 / (n - 1)
        d = max(number_of_samples -1 ,1)
        eigenvalues = (singular_val ** 2) / d
        tot_varience = np.sum(eigenvalues)
        self.varience = eigenvalues[:number_of_components]

        if tot_varience > 0: self.varience_ratio = self.varience / tot_varience
        else: self.varience_ratio = np.zeros(number_of_components,dtype=np.float32)
        return self

    def transform(self,X):
        if self.components is None : raise ValueError("Run fit before transforming the data")
        X = np.asarray(X,dtype=np.float32)
        projection = (X - self.mean_features).dot(self.components.T)
        return projection

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    

        


        


        
