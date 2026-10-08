import numpy as np
from .base import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self, lr=0.1, max_iter=1000, tol=1e-5, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _sigmoid(self, z):
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_design = self._add_intercept(X)
            weights = np.zeros((n_features + 1, 1), dtype=np.float32)
        else:
            X_design = X
            weights = np.zeros((n_features, 1), dtype=np.float32)

        for epoch in range(self.max_iter):
            linear_pred = X_design @ weights
            probs = self._sigmoid(linear_pred)
            error = probs - y
            grad = (X_design.T @ error) / n_samples
            weights -= self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = float(weights[0, 0])
            self.coef_ = weights[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.flatten()

        return self

    def predict_proba(self, X):
        z = self._decision_function(X)
        return self._sigmoid(z).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int32)