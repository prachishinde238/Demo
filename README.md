# Demo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

dataset = pd.read_csv('SalaryPosition.csv')
X = dataset.iloc[:, 1:2].values   # Level
y = dataset.iloc[:, 2].values     # Salary

lin_reg = LinearRegression()
lin_reg.fit(X, y)


poly_reg = PolynomialFeatures(degree=4)  # You can change degree
X_poly = poly_reg.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)


linear_r2 = r2_score(y, lin_reg.predict(X))
poly_r2 = r2_score(y, lin_reg_poly.predict(X_poly))

print(f"Simple Linear Regression R2 Score: {linear_r2:.4f}")
print(f"Polynomial Regression R2 Score: {poly_r2:.4f}")

if poly_r2 > linear_r2:
    print("Polynomial Regression fits more accurately.")
else:
    print("Simple Linear Regression fits more accurately.")

levels_to_predict = np.array([[11], [12]])
salary_linear = lin_reg.predict(levels_to_predict)
salary_poly = lin_reg_poly.predict(poly_reg.transform(levels_to_predict))

print("\nPredicted Salaries:")
print(f"Level 11 (Linear): {salary_linear[0]:.2f}")
print(f"Level 11 (Polynomial): {salary_poly[0]:.2f}")
print(f"Level 12 (Linear): {salary_linear[1]:.2f}")
print(f"Level 12 (Polynomial): {salary_poly[1]:.2f}")


plt.scatter(X, y, color='red', label='Actual Data')

# Linear Regression Line
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Regression')

# Polynomial Regression Line
X_grid = np.arange(min(X), max(X) + 1, 0.1).reshape(-1, 1)
plt.plot(X_grid, lin_reg_poly.predict(poly_reg.transform(X_grid)),
         color='green', label='Polynomial Regression')

plt.title('Salary Prediction')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
