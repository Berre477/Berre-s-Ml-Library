import numpy as np
from .base import LinearModel


class Ridge(LinearModel):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = float(alpha)

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y)
        n_features = X.shape[1]

        if self.fit_intercept:
            X_design = self._add_intercept(X)
            reg = np.eye(n_features + 1, dtype=np.float32) * self.alpha
            reg[0, 0] = 0.0
        else:
            X_design = X
            reg = np.eye(n_features, dtype=np.float32) * self.alpha
        A = X_design.T @ X_design + reg
        b = X_design.T @ y
        params = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = float(params[0, 0])
            self.coef_ = params[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = params.flatten()
        return self

    def predict(self, X):
        return self._decision_function(X).flatten()