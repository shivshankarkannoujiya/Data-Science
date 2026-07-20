import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "economic_index.csv"

df_index = pd.read_csv(data_path)

# print(df_index.head())


"""TODO: Drop unnecessary Cols"""
df_index.drop(columns=["Unnamed: 0", "year", "month"], inplace=True)
# print(df_index.head())

# print(df_index.isnull().sum()) # No null values

"""TODO: Visualization"""


import seaborn as sns

# sns.pairplot(df_index)
# plt.show()

# print(df_index.corr())

# plt.scatter(df_index["interest_rate"], df_index["unemployment_rate"], color="r")
# plt.xlabel("Interest_rate")
# plt.ylabel("unemployment_rate")
# plt.show()

"""TODO: Seprate Independent and Dependent features"""
X = df_index.iloc[:, :-1]
y = df_index.iloc[:, -1]

# print(X.head())
# print(y.head())

"""TODO: Train & Test split"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# sns.regplot(
#     x="interest_rate", y="index_price", data=df_index
# )  # Plot data and a linear regression model fit
# plt.show()


# sns.regplot(x="interest_rate", y="unemployment_rate", data=df_index)
# sns.regplot(x="index_price", y="unemployment_rate", data=df_index)
# plt.show()


"""TODO: Standard Scaling"""
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# print(X_train)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train, y_train)

"""TODO: Cross Validation"""
from sklearn.model_selection import cross_val_score

validation_score = cross_val_score(
    estimator=regression, X=X_train, y=y_train, scoring="neg_mean_squared_error", cv=3
)

average_mse = np.mean(validation_score)

# print("Validation_score(mse): ", validation_score)
# print("Avg(mse): ", average_mse)

"""TODO: Prediction"""
y_predict = regression.predict(X_test)
# print(y_predict)


"""TODO: Performance Metrics"""
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mse)

# print(mse)
# print(mae)
# print(rmse)


from sklearn.metrics import r2_score

score = r2_score(y_test, y_predict)
adjusted_r2_score = (1 - (1 - score) * len(y_test) - 1) / (
    len(y_test) - X_test.shape[1] - 1
)
# print(score)
# print(adjusted_r2_score)

"""TODO: Assumption"""
# plt.scatter(y_test, y_predict)  # Linear Relationship means model performing well
# plt.show()

residual_error = y_test - y_predict
# print(residual_error)

"""Plot residual error"""
# sns.displot(residual_error, kind="kde") # Normal Distribution = model is good
# plt.show()

"""TODO: Scatter plot with respect to Prediction & residuals"""
plt.scatter(
    y_predict, residual_error
)  # Data is distributed UNIFORMLY < does not follow any pattern >
# plt.show()


"""TODO: Using OLS"""
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()

print(model.summary())
