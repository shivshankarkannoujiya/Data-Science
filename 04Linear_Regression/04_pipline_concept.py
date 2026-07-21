import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def polynomial_regresssion(degree: int):
    x_new = np.linspace(-3, 3, 200).reshape(200, 1)

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()
    poly_regression = Pipeline([("poly_features", poly_features), ("lin_reg", lin_reg)])

    poly_regression.fit(x_train, y_train)
    y_pred_new = poly_regression.predict(x_new)

    """Plotting Prediction line"""
    plt.plot(x_new, y_pred_new, "r-", label=f"Degree {degree}", linewidth=2)
    plt.plot(x_train, y_train, "b.", markersize=8, label="Train")
    plt.plot(x_test, y_test, "g.", markersize=8, label="Test")

    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis([-4, 4, 0, 10])

    plt.show()

polynomial_regresssion(8)