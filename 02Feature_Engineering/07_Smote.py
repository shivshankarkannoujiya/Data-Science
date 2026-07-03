from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.90],
    random_state=12,
)

"""
X: independent = Variable used to explain or predict another variable < features >.
y: dependent = Variable we want to predict.
"""

df1 = pd.DataFrame(X, columns=["f1", "f2"])
df2 = pd.DataFrame(y, columns=["target"])

final_df = pd.concat([df1, df2], axis=1)
# print(final_df.head())
# print(final_df["target"].value_counts())
# 0    900  < Majority >
# 1    100  < Minority >

# plt.scatter(final_df["f1"], final_df["f2"], c=final_df["target"])
# plt.show()


""" NOTE: To apply SMOTE we need lib: `imblearn` """
from imblearn.over_sampling import SMOTE

# TODO: Transform the dataset using the SMOTE
over_sample = SMOTE()
X, y = over_sample.fit_resample(final_df[["f1", "f2"]], final_df["target"])

# print(X.shape)  # (1800, 2)
# print(y.shape)  # (1800,)

# print(len(y[y == 0]))
# print(len(y[y == 1]))


df1 = pd.DataFrame(X, columns=["f1", "f2"])
df2 = pd.DataFrame(y, columns=["target"])

oversampled_df = pd.concat([df1, df2], axis=1)

plt.scatter(oversampled_df["f1"], oversampled_df["f2"], c=oversampled_df["target"])
plt.show()
