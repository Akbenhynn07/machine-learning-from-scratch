import numpy as np
from src.linear_regression import LinearRegressionScratch

np.random.seed(0)

X = np.random.rand(100,1)
y = 4 + 3*X[:,0] + np.random.randn(100)

model = LinearRegressionScratch()

model.fit(X,y)

pred = model.predict(X[:5])

print("Predictions:",pred)
