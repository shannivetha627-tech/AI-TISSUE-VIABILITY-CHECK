import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge

# Sample Regression Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2, 4, 5, 4, 5, 7, 8, 9])

# Train Bayesian Regression
model = BayesianRidge()

model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Bayesian Regression")
plt.title("Bayesian Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()