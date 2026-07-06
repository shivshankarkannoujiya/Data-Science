"""
EDA And Feature Engineering Of Google Play Store Dataset

Problem statement. Today, 1.85 million different apps are available for users to download. Android users have even more from which to choose, with 2.56 million available through the Google Play Store. These apps have come to play a huge role in the way we live our lives today. Our objective is to find:

    The most popular category
    The app with the largest number of installs
    The app with the largest size, etc.
    Data Collection.

Steps We Are Going to Follow
    1. Data Cleaning
    2. Exploratory Data Analysis
    3. Feature Engineering
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "googleplaystore.csv"

df = pd.read_csv(data_path)
# print(df.head())
# print(df.shape) (10841, 13) (apps, features)
# print(df.info())
# print(df.describe())
"""
ONLY ONE NUMERICAL VALUE PRESENT
            Rating
count  9367.000000
mean      4.193338
std       0.537431
min       1.000000
25%       4.000000
50%       4.300000
75%       4.500000
max      19.000000
"""

# TODO: Missing values

# print(df.isnull().sum())
"""
OBSERVATION:
- The dataset has missing values
"""

# TODO: Catchup every cols find out some other information from it.
# NOTE: DATA CLEANING
# 1. Reviews
# df["Reviews"].astype("int"): 3.0M will thought error

# <>
# print(df["Reviews"].unique())
"""
[   '159',    '967',  '87510', '215644',    '167',    '178',  '36815',
  '13791',    '121',  '13880',
 ...
   '2036',  '56496', '376223',    '785',   '5775',    '885',  '88486',
    '603',   '1195', '398307']
"""

# print(df["Reviews"].str.isnumeric().sum())  # 10840
# print(df[~df["Reviews"].str.isnumeric()])

df_copy = df.copy()
df_copy = df_copy.drop(df_copy.index[10472])
# print(df_copy[~df_copy["Reviews"].str.isnumeric()]) # Nothing is there

""" NOTE: Now we can convert `Reviews` into `int` """

df_copy["Reviews"] = df_copy["Reviews"].astype("int")
# print(df_copy.info())


# 2. TODO: Size

# print(df_copy["Size"].unique())
"""
[  '19M',   '14M',  '8.7M',   '25M',  '2.8M',  '5.6M',   '29M',   '33M',
  '3.1M',   '28M',
 ...
  '467k',  '157k',   '44k',  '676k',   '67k',  '552k',  '885k', '1020k',
  '582k',  '619k']
"""

# print(df_copy["Size"].isnull().sum())

# Make the all values into K => 19000M => 19M
df_copy["Size"] = df_copy["Size"].str.replace("M", "000")
df_copy["Size"] = df_copy["Size"].str.replace("k", "")
df_copy["Size"] = df_copy["Size"].replace("Varies with device", np.nan)

df_copy["Size"] = df_copy["Size"].astype("float")

# print(df_copy.info())
# print(df_copy["Size"])

"""OBSERVATION: There is one: NaN need to replace"""

# TODO: Installs & Price

# print(df_copy["Installs"].unique())
"""
[       '10,000+',       '500,000+',     '5,000,000+',    '50,000,000+',
       '100,000+',        '50,000+',     '1,000,000+',    '10,000,000+',
         '5,000+',   '100,000,000+', '1,000,000,000+',         '1,000+',
   '500,000,000+',            '50+',           '100+',           '500+',
            '10+',             '1+',             '5+',             '0+',
              '0',           'Free']
"""

# print(df_copy["Price"].unique())
"""
[       '0',    '$4.99',    '$3.99',    '$6.99',    '$1.49',    '$2.99',
    '$7.99',    '$5.99',    '$3.49',    '$1.99',    '$9.99',    '$7.49',
    '$0.99',    '$9.00',    '$5.49',   '$10.00',   '$24.99',   '$11.99',
   '$79.99',   '$16.99',   '$14.99',    '$1.00',   '$29.99',   '$12.99',
    '$2.49',   '$10.99',    '$1.50',   '$19.99',   '$15.99',   '$33.99',
   '$74.99',   '$39.99',    '$3.95',    '$4.49',    '$1.70',    '$8.99',
    '$2.00',    '$3.88',   '$25.99',  '$399.99',   '$17.99',  '$400.00',
    '$3.02',    '$1.76',    '$4.84',    '$4.77',    '$1.61',    '$2.50',
    '$1.59',    '$6.49',    '$1.29',    '$5.00',   '$13.99',  '$299.99',
  '$379.99',   '$37.99',   '$18.99',  '$389.99',   '$19.90',    '$8.49',
    '$1.75',   '$14.00',    '$4.85',   '$46.99',  '$109.99',  '$154.99',
    '$3.08',    '$2.59',    '$4.80',    '$1.96',   '$19.40',    '$3.90',
    '$4.59',   '$15.46',    '$3.04',    '$4.29',    '$2.60',    '$3.28',
    '$4.60',   '$28.99',    '$2.95',    '$2.90',    '$1.97',  '$200.00',
   '$89.99',    '$2.56',   '$30.99',    '$3.61',  '$394.99',    '$1.26',
 'Everyone',    '$1.20',    '$1.04']
"""

chars_to_remove = ["+", ",", "$"]
cols_to_clean = ["Installs", "Price"]

for item in chars_to_remove:
    for cols in cols_to_clean:
        df_copy[cols] = df_copy[cols].str.replace(item, "")

# print(df_copy["Installs"].unique())
# print(df_copy["Price"].unique())

"""TODO: Convert to integer value"""
df_copy["Installs"] = df_copy["Installs"].astype("int")
df_copy["Price"] = df_copy["Price"].astype("float")

# print(df_copy.info())


# TODO: Last Updated
# print(df_copy["Last Updated"].unique())
"""
[   'January 7, 2018',   'January 15, 2018',     'August 1, 2018',
       'June 8, 2018',      'June 20, 2018',     'March 26, 2017',
     'April 26, 2018',      'June 14, 2018', 'September 20, 2017',
       'July 3, 2018',
 ...
  'November 23, 2015',      'June 17, 2012',  'February 27, 2015',
  'December 18, 2013',   'February 6, 2012',  'November 25, 2014',
       'May 19, 2016',   'January 20, 2014',  'February 16, 2014',
     'March 23, 2014']
"""

# print(pd.to_datetime(df_copy["Last Updated"]))
df_copy["Last Updated"] = pd.to_datetime(df_copy["Last Updated"])
"""
0       2018-01-07
1       2018-01-15
2       2018-08-01
3       2018-06-08
4       2018-06-20
"""

df_copy["Day"] = df_copy["Last Updated"].dt.day
df_copy["Month"] = df_copy["Last Updated"].dt.month
df_copy["Year"] = df_copy["Last Updated"].dt.year

""" NOTE: We can drop the `Last Updated` """
df_copy.drop(columns="Last Updated", inplace=True)
# print(df_copy.info())

df_copy.to_csv("data/google_cleaned.csv")