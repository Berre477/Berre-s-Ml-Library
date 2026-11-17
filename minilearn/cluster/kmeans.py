import numpy as np

class Kmeans:

    def __init__(self,n_clusters=8,max_iter = 300,tol=1e-4,random_state = None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        
        