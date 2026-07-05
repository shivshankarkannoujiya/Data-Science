"""
EDA With Red Wine Data
Data Set Information:

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

Attributes information

Input features 

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

OUTPUT VARIABLE
12. quality (score between 0 to 10)
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent

data_path = BASE_DIR / "WineQT.csv"

df = pd.read_csv(data_path)
# print(df.head())

# print(df.info()) # Print concise summary of datasets <percentile, SD ...>
"""
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1143 non-null   float64
 1   volatile acidity      1143 non-null   float64
 2   citric acid           1143 non-null   float64
 3   residual sugar        1143 non-null   float64
 4   chlorides             1143 non-null   float64
 5   free sulfur dioxide   1143 non-null   float64
 6   total sulfur dioxide  1143 non-null   float64
 7   density               1143 non-null   float64
 8   pH                    1143 non-null   float64
 9   sulphates             1143 non-null   float64
 10  alcohol               1143 non-null   float64
 11  quality               1143 non-null   int64  
 12  Id                    1143 non-null   int64
"""

#  print(df.describe()) # generate descriptive statistics
"""
       fixed acidity  volatile acidity  ...      quality           Id
count    1143.000000       1143.000000  ...  1143.000000  1143.000000
mean        8.311111          0.531339  ...     5.657043   804.969379
std         1.747595          0.179633  ...     0.805824   463.997116
min         4.600000          0.120000  ...     3.000000     0.000000
25%         7.100000          0.392500  ...     5.000000   411.000000
50%         7.900000          0.520000  ...     6.000000   794.000000
75%         9.100000          0.640000  ...     6.000000  1209.500000
max        15.900000          1.580000  ...     8.000000  1597.000000
"""

# print(df.shape) # (1143, 13): (records, cols)

# TODO: list all the cols name
# print(df.columns)
"""
Index(
['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality', 'Id'],

      dtype='str')
"""

# TODO: get unique values in quality col
#print(df["quality"].unique())
"""[5 6 7 4 8 3]"""

# TODO: Check is there any missing values in the dataset
# print(df.isnull().sum())
"""
NOTE: NO MISSING VALUE
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
Id                      0
"""

# TODO: Find out the duplicate records
# print(df[df.duplicated()])

# TODO: Remove the duplicate
df.drop_duplicates(inplace=True)
# print(df.shape) # (1143, 13)


# TODO: Correlation
# print(df.corr()) # Compute pairwise correlation of columns, excluding NA/null values.

# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), annot=True)
# plt.show()


#TODO: count values in each 
# print(df.quality.value_counts())
# df.quality.value_counts().plot(kind="bar")
# plt.xlabel("Wine Quality")
# plt.ylabel("Count")
# plt.show()

"""
NOTE: Conclusion => This is `imbalanced dataset <SMOTE: Can be used to balance the dataset>`
quality
5    483
6    462
7    143
4     33
8     16
3      6
"""

# TODO: See the distribution of each and every cols
# print(df.head())

# for column in df.columns:
#     sns.histplot(df[column], kde=True)
#     plt.show()

# sns.histplot(df["alcohol"], kde=True)
# plt.show()


# TODO: Univariate, Bivariate and Multivariate Analysis
# sns.pairplot(df)
# plt.show()


# TODO: Categorical Plot
# sns.catplot(x="quality", y="alcohol", data=df, kind="box")
# plt.show()

# TODO: scatter plot
sns.scatterplot(x="alcohol", y="pH", hue="quality", data=df)
plt.show()

