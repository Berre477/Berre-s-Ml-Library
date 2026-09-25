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

        for col in range(n_featues):
            if self.categories == "auto":
                cats = np.unique(X[:,col])

            else:
                cats = np.array(self.categories[col])

            self.categories_.append(cats)
            self.mapping_.append({cat: idx for idx,cat in enumerate(cats)})
        return self

    def transform(self,X):
        X = np.asarray(X)
        X_out = np.zeros(X.shape,dtype=np.float32)


        

class OneHotEncoder :
    def __init__(self,handle_unknown ='ignore'):
        self.categories_ = []
        self.handle_unknown = handle_unknown

    def fit(self,X):
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:,col]) for col in range(X.shape[1])]
        return self

    def transform(self,X):
        X = np.asarray(X)
        encoded_blocks = []

        for col_idx,categories in enumerate(self.categories_):
            col_data = X[:,col_idx]
            num_categories = len(categories)
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            one_hot_block = np.zeros((len(col_data), num_categories), dtype=np.float32)

            for row_idx,val in enumerate(col_data):
                if val in mapping:
                    one_hot_block[row_idx,mapping[val]] = 1.0
                elif self.handle_unknown == "errr":
                    raise ValueError(f"Unknown category {val} is in the column {col_idx}")

            encoded_blocks.append(one_hot_block)


        return np.hstack(encoded_blocks)
