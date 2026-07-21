import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = 6 * np.random.rand(100, 1) - 3  # independent feature
y = 0.5 * x**2 + 1.5 * x + 2 + np.random.rand(100, 1)  # dependent feature

"""Quadratic equation used: y = 0.5x^2 + 1.5x + 2 + outlier"""

# plt.scatter(x, y, color="g")
# plt.xlabel("x dataset")
# plt.ylabel("y dataset")
# plt.savefig("polynomial_reg.png")
# plt.show()


"""TODO: Train & Test split"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

"""Implement Linear regression"""
from sklearn.linear_model import LinearRegression

regression_1 = LinearRegression()
regression_1.fit(x_train, y_train)

from sklearn.metrics import r2_score

score = r2_score(y_test, regression_1.predict(x_test))
# print(score)

"""Visualization"""
# plt.plot(x_train, regression_1.predict(x_train), color="red")
# plt.scatter(x_train, y_train)
# plt.xlabel("X dataset")
# plt.ylabel("y dataset")

# plt.savefig("linear_regression_in_poly.png", bbox_inches="tight", dpi=300)
# plt.show()


"""TODO: Apply Polynomial Transformation"""
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# print(x_train_poly)

regression = LinearRegression()
regression.fit(x_train_poly, y_train)
y_pred = regression.predict(x_test_poly)
score = r2_score(y_test, y_pred)
# print(score)

# print(regression.coef_)
# print(regression.intercept_)

# plt.scatter(x_train, regression.predict(x_train_poly), color="red")
# plt.scatter(x_train, y_train)

# plt.savefig("poly_regression.png", bbox_inches="tight", dpi=300)
# plt.show()


"""TODO: Increase Degree"""
poly = PolynomialFeatures(degree=3, include_bias=True)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# print(x_train_poly)
regression = LinearRegression()
regression.fit(x_train_poly, y_train)
y_pred = regression.predict(x_test_poly)
score = r2_score(y_test, y_pred)
# print(score)

# plt.scatter(x_train, regression.predict(x_train_poly), color="red")
# plt.scatter(x_train, y_train)

# plt.savefig("poly_regression_degree-3.png", bbox_inches="tight", dpi=300)
# plt.show()


"""TODO: Prediction for new data"""
x_new = np.linspace(-3, 3, 200).reshape(200, 1)
x_new_poly = poly.transform(x_new)

y_new = regression.predict(x_new_poly)
plt.plot(x_new, y_new, "r-", linewidth=2, label="New Prediction")
plt.plot(x_train, y_train, "b.", label="Training Points")
plt.plot(x_test, y_test, "g.", label="Testing points")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig(
    "poly_regression_degree-3_predict_new_data.png", bbox_inches="tight", dpi=300
)
plt.show()
