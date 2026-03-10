import numpy as np
from gradient_descent import gradient_descent

class LinearRegressionScratch:

    def fit(self, X, y):
        self.w, self.b = gradient_descent(X, y)

    def predict(self, X):
        return X @ self.w + self.b
