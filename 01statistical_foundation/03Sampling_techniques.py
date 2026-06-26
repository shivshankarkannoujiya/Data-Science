import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "Population_Survey_Data_lyst1750274390196.csv"

data = pd.read_csv(csv_path)
# print(data.head())
# print(data.shape)


"""
TODO: Simple random Technique
Fetch 100 records randomly
"""

sample_data = data.sample(n=100, random_state=44)
# print(sample_data)


"""
TODO: Stratified Random sampling
Want 25 samples from each region
if not available 25 sample in any region eg: have 15 then 
Return the min available sample
"""

stratified_sample_data = data.groupby("Region").apply(
    lambda x: x.sample(min(len(x), 25), random_state=44)
)

# stratified_sample_data.to_csv("stratified_sample_data.csv")
# print(stratified_sample_data)


"""
TODO: Cluster Sampling: Manufacturing data by BatchNumber.
create cluster by: BatchNumber
Eg: 25, 30 
"""

manufacturing_data_path = BASE_DIR / "Manufacturing_Data_lyst1750274395200.csv"
manufacturing_data = pd.read_csv(manufacturing_data_path)

cluster = manufacturing_data["BatchNumber"].unique()
# print(cluster)

"""
[25 45 20 31 14  8 36 30 21  9 11 42 35 47 12 22  5 48 23  7 44 33 38  4
 18  6 28  2 34 41 40 19 16 46 49 43 17  3 13 10 29 32 24 26 39 15  1 27
 37]

NOTE: Cluster available in list
So we can use `choice`
"""

selected_cluster = np.random.choice(cluster, size=5, replace=False)
# print(selected_cluster)

clustered_sample = manufacturing_data[
    manufacturing_data["BatchNumber"].isin(selected_cluster)
]

# clustered_sample.to_csv("clustered_sample.csv")
# print(clustered_sample)


"""
TODO: Systematic Sampling
"""

systematic_sample = data.iloc[::10, :]
# print(systematic_sample)

"""
TODO: 
- Evaluate the measures of central tendency for the whole data
- Followed my simple random sample, stratified random sample, cluster sample, systematic random sample
- You need to compare Means of all the different types of sample
- Finally conclude which specific sampling technique Mean value is almost similar to the  Population Mean value
"""
