from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "height-weight.csv"

df = pd.read_csv(data_path)
# print(df.head())

"""TODO: Check how Independent and dependent features are related"""
# Scatter plot
# plt.scatter(df["Weight"], df["Height"])
# plt.xlabel("Weight")
# plt.ylabel("Height")
# plt.savefig("ind_dep_relation.png", dpi=300, bbox_inches="tight")
# plt.show()

"""
OBSERVATION:
    - As the Weight increases Height is also increasing: Linear Relationship
NOTE:
To check whether this relationship is positive or negative: < Correlation >
"""
# print(df.corr())

"""
        Weight    Height
Weight  1.000000  0.931142
Height  0.931142  1.000000
NOTE: POSITIVE
"""

# sns.pairplot(df)
# plt.show()

""" 
Independent and dependent features
    Weight: Independent
    Height: Dependent
"""

# x = df["Weight"]
# print(type(x)) <class 'pandas.Series'>
# NOTE: We want DataFrame

x = df[["Weight"]]
# print(type(x)) <class 'pandas.DataFrame'>

"""
NOTE: Independent feature should be 
    - DataFrame or
    - 2D array
NOTE: Dependent feature can be the: Series < One col value > and 1D
"""
y = df["Height"]


"""TODO: Split data: TRAIN & TEST"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# print(x_train.shape)


""" 
TODO: Standardization < Input feature >
    - Take each independent features
    - Apply Z-score: it convert all the values with
        - Mean = 0 & Standard Deviation = 1
"""
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
# print(x_train)

x_test = scalar.transform(x_test)
# print(x_test)


"""TODO: Apply Simple Linear Regression"""
from sklearn.linear_model import LinearRegression

regression = LinearRegression(n_jobs=-1)

regression.fit(x_train, y_train)

# print(
#     "Coefficient:(Slope)", regression.coef_
# )  # [17.2982057] """ 1 unit movement in the weight value that leads to 17.2982057 movement in the Height value """
# print("Intercept:", regression.intercept_)  # 156.47058823529412


"""Plot bestfit line  with respect to Training Data"""

# plt.scatter(x_train, y_train)
# plt.plot(x_train, regression.predict(x_train))
# plt.savefig("bestfit_line_respect_to_X_train.png", dpi=300, bbox_inches="tight")
# plt.show()


"""
NOTE: Prediction of test data
    1. Predicted height output = intercept + coef_(Weight)
    2. y_pred_test = 156.470 + 17.29(x_test)
"""

"""TODO: Prediction for test_data"""
y_pred = regression.predict(x_test)
# print(y_pred)
"""
[162.26499721 162.26499721 127.68347133 180.07972266 148.64197186
 190.55897293]
"""


"""Performance Metrics"""
from sklearn.metrics import mean_absolute_error, mean_squared_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# print("MSE: ", mse)
# print("MAE: ", mae)
# print("RMSE: ", rmse)

"""
MSE:  114.84069295228699
MAE:  9.66512588679501
RMSE:  10.716374991212605
"""


from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)
# print("r2_score: ", score) # r2_score:  0.7360826717981276

adjusted_r2_score = 1 - (1 - score) * (len(y_test) - 1) / (
    len(y_test) - x_test.shape[1] - 1
)
# print(adjusted_r2_score) # 0.6701033397476595

"""OLS Linear Regression"""
import statsmodels.api as sm

model = sm.OLS(y_train, x_train).fit()
predicted_value = model.predict(x_test)
# print(predicted_value)
"""
[  5.79440897   5.79440897 -28.78711691  23.60913442  -7.82861638
  34.08838469]
"""

# TODO: Print model summary
# print(model.summary())

"""Prediction for new value"""
print(regression.predict(scalar.transform([[72]])))
