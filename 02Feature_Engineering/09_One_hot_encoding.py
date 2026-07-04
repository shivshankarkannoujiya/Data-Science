import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create simple dataframe
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

# TODO: Create instance of OneHotEncoder

encoder = OneHotEncoder()

# Perform fit and then transform
# print(encoder.fit_transform(df[["color"]])) # create sparse matrix

# Convert into array
# print(encoder.fit_transform(df[["color"]]).toarray())
"""
NOTE: Sorted Alphabetically
  b. g. r
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
"""

encoded_values = encoder.fit_transform(df[["color"]]).toarray()

encoded_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out())
print(encoded_df)

"""
   color_blue  color_green  color_red
0         0.0          0.0        1.0
1         1.0          0.0        0.0
2         0.0          1.0        0.0
3         0.0          1.0        0.0
4         0.0          0.0        1.0
5         1.0          0.0        0.0
"""

# New data that come should be only red. blue, green only
# print(encoder.transform([["blue"]]).toarray())

data = pd.concat([df, encoded_df], axis=1)
print(data)

"""
   color  color_blue  color_green  color_red
0    red         0.0          0.0        1.0
1   blue         1.0          0.0        0.0
2  green         0.0          1.0        0.0
3  green         0.0          1.0        0.0
4    red         0.0          0.0        1.0
5   blue         1.0          0.0        0.0
"""