import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "size": ["small", "medium", "large", "medium", "small", "large"]
})

# print(df)
"""
     size
0   small
1  medium
2   large
3  medium
4   small
5   large
"""

ordinal_encode = OrdinalEncoder(categories=[["small", "medium", "large"]])
encoded_values = ordinal_encode.fit_transform(df[["size"]])

# print(encoded_values)
"""
[[0.]
 [1.]
 [2.]
 [1.]
 [0.]
 [2.]]
"""

# New values
# print(ordinal_encode.transform([["small"]])) # [[0.]]