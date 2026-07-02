import seaborn as sns

"""
- Use: Categorical Values
"""

data = sns.load_dataset('titanic')
# print(data[data["embarked"].isnull()])
# print(data["embarked"].unique())

mode_value = data[data["embarked"].notna()]["embarked"].mode()[0]
print(mode_value)
data["embarked_mode"] = data["embarked"].fillna(mode_value)

print(data[["embarked_mode", "embarked"]])
print(data["embarked_mode"].isnull().sum()) # 0
print(data["embarked"].isnull().sum()) # 2