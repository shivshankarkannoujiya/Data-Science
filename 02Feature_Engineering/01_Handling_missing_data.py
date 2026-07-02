import seaborn as sns

df = sns.load_dataset("titanic")
# print(df.head())

# check missing values
# print(df.isnull().sum())
# print(df.shape) # (891, 15)

# TODO: Delete the rows that have missing values 
"""
Problem: We will loose huge amt of data
"""
# NOTE: loose a lot of data
# print(df.dropna().shape) # (182, 15)

"""Can we remove Column wise: axis = 1, inplace = True <To make permanent>"""
# print(df.dropna(axis=1))

"""
DROPPED THE Age and embark colm.
     survived  pclass     sex  sibsp  parch     fare   class    who  adult_male alive  alone
0           0       3    male      1      0   7.2500   Third    man        True    no  False
1           1       1  female      1      0  71.2833   First  woman       False   yes  False
2           1       3  female      0      0   7.9250   Third  woman       False   yes   True
3           1       1  female      1      0  53.1000   First  woman       False   yes  False
4           0       3    male      0      0   8.0500   Third    man        True    no   True
..        ...     ...     ...    ...    ...      ...     ...    ...         ...   ...    ...
886         0       2    male      0      0  13.0000  Second    man        True    no   True
887         1       1  female      0      0  30.0000   First  woman       False   yes   True
888         0       3  female      1      2  23.4500   Third  woman       False    no  False
889         1       1    male      0      0  30.0000   First    man        True   yes   True
890         0       3    male      0      0   7.7500   Third    man        True    no   True
"""

