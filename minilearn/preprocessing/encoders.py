import numpy as np

class LabelEnoder:
    def __init__(self):
        self.classes_ = None
        self._mapping = None
        self._imapping = None

    def fit(self,y):
        self.classes_ = np.unique(y)
        self._mapping = {label: idx for idx, label in enumerate(self.classes_)}
        self._inv_mapping = {idx: label for label, idx in self._mapping.items()}
        return self

    def transform(self,y):
        return np.array([self._mapping[label] for label in y], dtype=np.int64)

    def fit_transform(self,y):
        return self.fit(y).transform(y)

    def inverse_transform(self,y_indices):
        return np.array([self._inv_mapping[idx] for idx in y_indices])




class OrdinalEncoder:

    def __init__(self,categories="auto"):
        self.categories = categories
        self.categories_ = []
        self.mapping_ = []

    def fit(self,X):
        X = np.asarray(X)
        n_featues = X.shape[1]
        self.categories_ = []
        self.mapping_ = []

    def transform(self,X):
        X = np.asarray(X)
        X_out = np.zeros(X.shape,dtype=np.float32)
        
        
        