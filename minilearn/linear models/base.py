import numpy as np

class LinearModel :

    def __init(self,fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    @classmethod
    def _prepare_inputs(self,X,u=None):

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            return X, y

        return X

    #Prepends a column of ones for the bias/intercept term
    def _add_intercept(self, X):
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1), dtype=np.float32)
        return np.hstack([ones, X])

    #Computes the raw linear score
    def _decision_function(self, X):
        X = self._prepare_inputs(X)
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return X @ self.coef_.reshape(-1, 1) + self.intercept_
        