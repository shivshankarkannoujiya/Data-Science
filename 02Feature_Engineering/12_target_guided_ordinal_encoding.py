
"""
- It is a technique used to encode categorical variables based on the relationship of the target variable
- This encoding technique is useful when we have categorical variable with a large number of unique categories and we want to use this variable as a feature in our machine learning model.

- In this we replace each category in the categorical variable with a numerical value based on the mean or median of the target variable for the category.
- This create a monotonic relationship between the categorical variable and the target variable, which can improve the predictive power of our model.
"""

import pandas as pd

df = pd.DataFrame({
    "city": ["New York", "London", "Paris", "Tokyo", "New York", "Paris"],
    "price": [200, 150, 300, 250, 180, 350]
})

"""Price: Target Varible"""

# print(df)
"""
    city      price
0  New York    200
1    London    150
2     Paris    300
3     Tokyo    250
4  New York    180
5     Paris    350
"""

# print(df.groupby("city")["price"].mean().to_dict())

"""
city
London      150.0
New York    190.0
Paris       325.0
Tokyo       250.0
Name: price, dtype: float64
"""
"""Convert to dictionary = {'London': 150.0, 'New York': 190.0, 'Paris': 325.0, 'Tokyo': 250.0}"""

mean_price = df.groupby("city")["price"].mean().to_dict()

df["city_encoded"] = df["city"].map(mean_price)
# print(df)

"""
       city  price  city_encoded
0  New York    200         190.0
1    London    150         150.0
2     Paris    300         325.0
3     Tokyo    250         250.0
4  New York    180         190.0
5     Paris    350         325.0
"""

# Now we can give price and city_encoded for training

print(df[["price", "city_encoded"]])
"""
   price  city_encoded
0    200         190.0
1    150         150.0
2    300         325.0
3    250         250.0
4    180         190.0
5    350         325.0
"""
