import numpy as np
import pandas as pd
from sklearn.utils import resample

np.random.seed(123)

# TODO: Create a DataFrame with two classes

n_samples = 1000

class_0_ratio = 0.9  # 90% class_0 ratio

n_class_0 = int(n_samples * class_0_ratio)  # 1000 * 0.9 = 900 datapoints
n_class_1 = n_samples - n_class_0

class_0 = pd.DataFrame(
    {
        "feature_1": np.random.normal(loc=0, scale=1, size=n_class_0),
        "feature_2": np.random.normal(loc=0, scale=1, size=n_class_0),
        "target": [0] * n_class_0,  # 900 0 will get created
    }
)

class_1 = pd.DataFrame(
    {
        "feature_2": np.random.normal(loc=2, scale=1, size=n_class_1),
        "feature_1": np.random.normal(loc=2, scale=1, size=n_class_1),
        "target": [1] * n_class_1,  # 100 1 will get created
    }
)

data = pd.concat([class_0, class_1]).reset_index(drop=True)
# print(data.head())
# print(data.tail())
# print(data["target"].value_counts())
#  # 0    900
#  1    100


# TODO: Down Sampling

data_minority = data[data["target"] == 1]
data_majority = data[data["target"] == 0]

majority_down_sampled = resample(
    data_majority, replace=False, n_samples=len(data_minority), random_state=42
)

print(majority_down_sampled.shape)  # (900, 3) --- Down Sampled ---> (100, 3)

data_downsampled = pd.concat([data_minority, majority_down_sampled])
print(data_downsampled.target.value_counts())
# 1    100
# 0    100

"""NOTE: Down sampling is Bad, because we are loosing the data_points"""
