import seaborn as sns
import matplotlib.pyplot as plt

"""
- Replace the missing value with the mean of that specific column
- Mean Imputation works when we have `Normally Distributed Data`

"""

data = sns.load_dataset("titanic")

# sns.histplot(data["age"], kde=True)
# plt.show()

data["Age_mean"] = data["age"].fillna(data["age"].mean())
print(data[["Age_mean", 'age']])


# NOTE: what if we have right/left skewed data: There will be outliers
# TODO: In this case replace NaN value with `Median`
