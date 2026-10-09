import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape

        if self.n_components is None:
            k = min(n_samples, n_features)
        else:
            k = min(self.n_components, n_samples, n_features)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        max_abs_cols = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs_cols])
        Vt *= signs[:, np.newaxis]
        self.components_ = Vt[:k]

        total_variance = np.sum((S ** 2) / (n_samples - 1))
        self.explained_variance_ = ((S[:k] ** 2) / (n_samples - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("PCA is not fited.")
        X = np.asarray(X, dtype=np.float32)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        if self.components_ is None:
            raise RuntimeError("PCA is not fited.")
        X_transformed = np.asarray(X_transformed, dtype=np.float32)
        return (X_transformed @ self.components_) + self.mean_