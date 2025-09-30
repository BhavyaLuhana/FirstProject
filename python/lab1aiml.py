import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

seed = np.random.seed(0)
x = 2 * np.random.rand(200, 1)
print(x)
y = 4+3*x + np.random.randn(200, 1)
print(y)

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, color='red', label='Actual data')
plt.plot(x_test, y_pred, color='green', label='Prediction line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()