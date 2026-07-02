import seaborn as sns

"""
- NOTE: what if we have right/left skewed data: There will be `Outliers`
- TODO: In this case replace NaN value with `Median`
"""

data = sns.load_dataset('titanic')

data["age_median"]= data["age"].fillna(data['age'].median())
print(data[["age_median", "age"]])

