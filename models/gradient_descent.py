import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for i in range(epochs):

    y_pred = m*X + b

    dm = (-2/n) * np.sum(X*(Y-y_pred))
    db = (-2/n) * np.sum(Y-y_pred)

    m = m - learning_rate*dm
    b = b - learning_rate*db

print("Slope:",m)
print("Intercept:",b)

plt.scatter(X,Y,color="blue")
plt.plot(X,m*X+b,color="red")
plt.title("Gradient Descent")
plt.show()