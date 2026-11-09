import numpy as np

class  KNClassifier:
    def __init__(self,n_neighbors = 10):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.classes = None