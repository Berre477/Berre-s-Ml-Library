import numpy as np
from .base import LinearModel


class LinearRegression(LinearModel):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y)

        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X

        A = X_design.T @ X_design
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