import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""Assign Unique numerical value to every categories"""

df = pd.DataFrame({
    "color": ["red", "blue", "green", "green", "red", "blue"]
})

# print(df.head())
"""
   color
0    red
1   blue
2  green
3  green
4    red
"""

lable_encode = LabelEncoder()
encoded_value = lable_encode.fit_transform(df[["color"]])

# print(encoded_value) # [2 0 1 1 2 0]

# print(lable_encode.transform([["red"]])) # [2]
# print(lable_encode.transform([["green"]])) # [1]
# print(lable_encode.transform([["blue"]]))  # [0]


