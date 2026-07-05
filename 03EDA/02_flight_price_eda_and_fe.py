"""
----- FLIGHT PRICE PREDICTION -----

These are the main Features/Columns available in the dataset :

1) Airline: The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.

2) Flight: Flight stores information regarding the plane's flight code. It is a categorical feature.

3) Source City: City from which the flight takes off. It is a categorical feature having 6 unique cities.

4) Departure Time: This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and have 6 unique time labels.

5) Stops: A categorical feature with 3 distinct values that stores the number of stops between the source and destination cities.

6) Arrival Time: This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.

7) Destination City: City where the flight will land. It is a categorical feature having 6 unique cities.

8) Class: A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.

9) Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.

10) Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.

11) Price: Target variable stores information of the ticket price.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "flight_price.xlsx"

df = pd.read_excel(data_path)
# print(df.head())
# print(df.tail())

# TODO: 1. Get the basic info about the data
# print(df.info())

"""
RangeIndex: 10683 entries, 0 to 10682
Data columns (total 11 columns):
#   Column           Non-Null Count  Dtype
---  ------           --------------  -----
0   Airline          10683 non-null  str
1   Date_of_Journey  10683 non-null  str
2   Source           10683 non-null  str
3   Destination      10683 non-null  str
4   Route            10682 non-null  str
5   Dep_Time         10683 non-null  str
6   Arrival_Time     10683 non-null  str
7   Duration         10683 non-null  str
8   Total_Stops      10682 non-null  str
9   Additional_Info  10683 non-null  str
10  Price            10683 non-null  int64

NOTE:
- A Pandas column is a Series, not a Python string.
- Use the .str accessor to apply string methods to every element in the column.
- We need to typecast if want to convert to int or float
"""

# print(df.describe())
"""
BECAUSE ONLY One Numerical feature
Price
count  10683.000000
mean    9087.064121
std     4611.359167
min     1759.000000
25%     5277.000000
50%     8372.000000
75%    12373.000000
max    79512.000000
"""

# Date_of_journey
"""We can split: day month year easily <string data type>"""
# print(df["Date_of_Journey"].str.split("/").str[0])

df["Date"] = df["Date_of_Journey"].str.split("/").str[0]
df["Month"] = df["Date_of_Journey"].str.split("/").str[1]
df["Year"] = df["Date_of_Journey"].str.split("/").str[2]

# print(df.head())
"""
Airline Date_of_Journey    Source Destination  ...  Price Date Month  Year
0       IndiGo      24/03/2019  Banglore   New Delhi  ...   3897   24    03  2019
1    Air India       1/05/2019   Kolkata    Banglore  ...   7662    1    05  2019
2  Jet Airways       9/06/2019     Delhi      Cochin  ...  13882    9    06  2019
3       IndiGo      12/05/2019   Kolkata    Banglore  ...   6218   12    05  2019
4       IndiGo      01/03/2019  Banglore   New Delhi  ...  13302   01    03  2019

New cols are reflected.
But,
They are still `object` type
"""
# print(df.info())
"""
#   Column           Non-Null Count  Dtype
---  ------           --------------  -----
0   Airline          10683 non-null  str
1   Date_of_Journey  10683 non-null  str
2   Source           10683 non-null  str
3   Destination      10683 non-null  str
4   Route            10682 non-null  str
5   Dep_Time         10683 non-null  str
6   Arrival_Time     10683 non-null  str
7   Duration         10683 non-null  str
8   Total_Stops      10682 non-null  str
9   Additional_Info  10683 non-null  str
10  Price            10683 non-null  int64

11  Date             10683 non-null  object
12  Month            10683 non-null  object
13  Year             10683 non-null  object
"""

# TODO: Convert into the Numerical

df["Date"] = df["Date"].astype("int")
df["Month"] = df["Month"].astype("int")
df["Year"] = df["Year"].astype("int")

# print(df.info())
"""
NOTE: Converted into numerical
11  Date             10683 non-null  int64
12  Month            10683 non-null  int64
13  Year             10683 non-null  int64
"""

"""
--- Since i have converted `Date_of_Journey` into Data, Month, Year ---
- I can drop this col.
"""
# TODO: Drop Date_of_Journey col
df.drop("Date_of_Journey", axis=1, inplace=True)
# print(df.head())


# TODO: 2. Similarly: Arrival_Time

# print(df["Arrival_Time"])
"""
0        01:10 22 Mar
1               13:15
2        04:25 10 Jun
3               23:30
4               21:35
...
10678           22:25
10679           23:20
10680           11:20
10681           14:10
10682           19:15
"""

df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: x.split(" ")[0])
# print(df["Arrival_Time"])

"""
0        01:10
1        13:15
2        04:25
3        23:30
4        21:35
...
10678    22:25
10679    23:20
10680    11:20
10681    14:10
10682    19:15
"""

df["Arrival_hour"] = df["Arrival_Time"].str.split(":").str[0]
df["Arrival_minute"] = df["Arrival_Time"].str.split(":").str[1]

# print(df["Arrival_hour"])
"""
0        01
1        13
2        04
3        23
4        21
..
10678    22
10679    23
10680    11
10681    14
10682    19
"""

# print(df["Arrival_minute"])
"""
0        10
1        15
2        25
3        30
4        35
..
10678    25
10679    20
10680    20
10681    10
10682    15
"""
# TODO: Arrival_hour, Arrival_minute convert into integer

df["Arrival_hour"] = df["Arrival_hour"].astype("int")
df["Arrival_minute"] = df["Arrival_minute"].astype("int")

# TODO: Drop Arrival_Time
df.drop("Arrival_Time", axis=1, inplace=True)

# print(df.head(2))
"""
NOTE: Arrival_Time Deleted

Airline    Source Destination                  Route  ... Month  Year Arrival_hour Arrival_minute
0     IndiGo  Banglore   New Delhi              BLR → DEL  ...     3  2019            1             10
1  Air India   Kolkata    Banglore  CCU → IXR → BBI → BLR  ...     5  2019           13             15
"""

# TODO: handle Dept_Time

# print(df["Dep_Time"])
"""
0        22:20
1        05:50
2        09:25
3        18:05
4        16:50
...
10678    19:55
10679    20:45
10680    08:20
10681    11:30
10682    10:55
"""

df["Dep_hour"] = df["Dep_Time"].str.split(":").str[0]
df["Dep_minute"] = df["Dep_Time"].str.split(":").str[1]

# print(df["Dep_hour"])
"""
0        22
1        05
2        09
3        18
4        16
..
10678    19
10679    20
10680    08
10681    11
10682    10
Name: Dep_hour, Length: 10683, dtype: object
"""
# print(df["Dep_minute"])
"""
0        20
1        50
2        25
3        05
4        50
..
10678    55
10679    45
10680    20
10681    30
10682    55
Name: Dep_minute, Length: 10683, dtype: object
"""


# TODO: Convert into Numerical

df["Dep_hour"] = df["Dep_hour"].astype("int")
df["Dep_minute"] = df["Dep_minute"].astype("int")

# TODO:  Now drop the col <Dep_Time>

df.drop("Dep_Time", axis=1, inplace=True)
# print(df.head(2))

# print(df.info())


# TODO: Handle Categorical Data
# print(df["Total_Stops"].unique())
""" ['non-stop', '2 stops', '1 stop', '3 stops', nan, '4 stops'] """

# TODO: nan: Missing value
# print(df[df["Total_Stops"].isnull()])

"""         Airline Source Destination Route Duration  ...  Year Arrival_hour  Arrival_minute  Dep_hour  Dep_minute
9039  Air India  Delhi      Cochin   NaN  23h 40m  ...  2019            9              25         9          45 """

# print(df["Total_Stops"].mode()) # 0    1 stop

df["Total_Stops"] = df["Total_Stops"].map(
    {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4, np.nan: 0}
)

# print(df[df["Total_Stops"].isnull()]) No nan exists

"""
NOTE: We can drop `Route` because we have 2 seprate features
- Source
- Destination
"""
df.drop("Route", axis=1, inplace=True)
# print(df.head())


# TODO: Handle `Duration`
df["Duration_hour"] = df["Duration"].str.split(" ").str[0].str.split("h").str[0]
df["Duration_minute"] = df["Duration"].str.split(" ").str[1].str.split("m").str[0]
# print(df["Duration_hour"])
# print(df["Duration_minute"])

# TODO: Handle `Duration_minute` NaN value


def split_duration(duration):
    hour = 0
    minute = 0

    for part in duration.split():
        if part.endswith("h"):
            hour = int(part[:-1])
        elif part.endswith("m"):
            minute = int(part[:-1])
    return hour, minute


df[["Duration_hour", "Duration_minute"]] = (
    df["Duration"].apply(split_duration).apply(pd.Series)
)

df.drop(columns=["Duration"], inplace=True)
# print(df["Duration_hour"])
# print(df["Duration_minute"])


"""TODO: Handle Airline"""
# print(df["Airline"].unique())
"""
<StringArray>
[ 
'IndiGo','Air India','Jet Airways','SpiceJet','Multiple,carriers','GoAir','Vistara','Air Asia','Vistara Premium economy','Jet Airways Business','Multiple carriers Premium, economy','Trujet'
]

NOTE: Perform One Hot Encoding
"""

# print(df["Source"].unique())  # ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
# print(df["Destination"].unique())  # ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']

# print(df["Additional_Info"].unique())
"""
[                     'No info',  'In-flight meal not included',
 'No check-in baggage included',              '1 Short layover',
                      'No Info',               '1 Long layover',
              'Change airports',               'Business class',
               'Red-eye flight',               '2 Long layover']
"""

# TODO: Convert this categorical feature into Numerical feature using OHE

encoder = OneHotEncoder(sparse_output=False)

# NOTE: if not given: sparse_output=False
# encoded_value = encoder.fit_transform(df[["Airline", "Source", "Destination"]]).toarray()
# print(encoded_value)

encoded = encoder.fit_transform(df[["Airline", "Source", "Destination"]])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Airline", "Source", "Destination"]),
    index=df.index,
)

# print(encoded_df)
"""
       Airline_Air Asia  Airline_Air India  ...  Destination_Kolkata  Destination_New Delhi
0                   0.0                0.0  ...                  0.0                    1.0
1                   0.0                1.0  ...                  0.0                    0.0
2                   0.0                0.0  ...                  0.0                    0.0
3                   0.0                0.0  ...                  0.0                    0.0
4                   0.0                0.0  ...                  0.0                    1.0
...                 ...                ...  ...                  ...                    ...
10678               1.0                0.0  ...                  0.0                    0.0
10679               0.0                1.0  ...                  0.0                    0.0
10680               0.0                0.0  ...                  0.0                    0.0
10681               0.0                0.0  ...                  0.0                    1.0
10682               0.0                1.0  ...                  0.0                    0.0
"""


"""TODO: Merge it with the original DataFrame"""
df = pd.concat([df, encoded_df], axis=1)

# print(df.head())

# No longer need the original categorical columns:
df.drop(columns=["Airline", "Source", "Destination"], inplace=True)

# print(df.head())


# TODO: Check the remaining categorical columns
# print(df.dtypes[df.dtypes == "object"])
# print(df.dtypes)

"""TODO: Handle Additional_Info"""
# print(df["Additional_Info"].unique())
# print(df["Additional_Info"].value_counts())

# Step 1. Understand the feature
# print(df["Additional_Info"].head())
"""
Ask:
    What does this column represent?
    Is it categorical?
    Is it ordinal?
    Is it useful?
"""

# Step 2. Check missing values
# print(df["Additional_Info"].unique())

"""
You already found

No info
No Info
In-flight meal not included
Business class
...
"""


# Step 4. Check cardinality
# print(df["Additional_Info"].nunique())
""" 10: Good candidate for One-Hot Encoding. """

# Step 5. Check frequency
# print(df["Additional_Info"].value_counts())

"""
Additional_Info
No info                         8345
In-flight meal not included     1982
No check-in baggage included     320
1 Long layover                    19
Change airports                    7
Business class                     4
No Info                            3
1 Short layover                    1
Red-eye flight                     1
2 Long layover                     1
Name: count, dtype: int64
"""

"""
This tells you
    common categories
    rare categories
    data quality issues
"""


# Step 6. Data Cleaning
"""
Immediately you notice
    - No info
    - No Info
These are the same thing.
Clean them.
"""

df["Additional_Info"] = (
    df["Additional_Info"].str.strip().str.lower()
)  # no info <becomes one category.>

# Step 7. Rare Category Analysis
print(df["Additional_Info"]
    .value_counts(normalize=True)
    *100
)

"""
Additional_Info
no info                         78.142844
in-flight meal not included     18.552841
no check-in baggage included     2.995413
1 long layover                   0.177853
change airports                  0.065525
business class                   0.037443
1 short layover                  0.009361
red-eye flight                   0.009361
2 long layover                   0.009361
Name: proportion, dtype: float64
"""
"""
NOTE:
no info                    78%
meal not included          18%
everything else <1%

Now ask
    - Should I keep them?
"""

# Step 8. Group Rare Categories
# A common threshold is <1%.

freq = df["Additional_Info"].value_counts(normalize=True)
rare = freq[freq < 0.01].index
# print(freq)
# print(rare)


# Replace
df["Additional_Info"] = df["Additional_Info"].replace(rare, "other")

"""
Now,
    no info
    meal not included
    no baggage
    other
"""

# Step 9. Encode
encoder = OneHotEncoder(sparse_output=False, dtype=int)
encoded = encoder.fit_transform(df[["Additional_Info"]])

encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(["Additional_Info"]), index=df.index
)

df = pd.concat([df, encoded_df], axis=1)

# Step 10. Drop original column
df.drop("Additional_Info", axis=1, inplace=True)
