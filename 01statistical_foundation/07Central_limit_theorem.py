import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent

data_path = BASE_DIR / "fraud_data_lyst1751134022071.xlsx"

data = pd.read_excel(data_path)
# print(data.head())

# 1. Get the population mean
population_mean = data["amt"].mean()
# print(f"Population mean: {population_mean}")  # 124.43007337671327

# 2. Get the sample mean
sample = data["amt"].sample(n=1000, random_state=42)
sample_mean = sample.mean()
# print(f"Sample mean: {sample_mean}")  # 119.54582999999998


# Impact of sample size on sample mean
sample_sizes = [10, 50, 100, 500, 1000, 5000, 7000]

sample_means = [
    data["amt"].sample(n=size, random_state=42).mean() for size in sample_sizes
]

for size, mean in zip(sample_sizes, sample_means):
    print(f"sample size: {size}, sample mean: {mean} ")

"""
sample size: 10, sample mean: 96.85499999999999 
sample size: 50, sample mean: 88.3492 
sample size: 100, sample mean: 81.8355 
sample size: 500, sample mean: 112.06866000000001 
sample size: 1000, sample mean: 119.54582999999998 
sample size: 5000, sample mean: 123.99536800000001 
sample size: 7000, sample mean: 126.52721
"""

# OBSERVATION
# NOTE:
# As the sample size increases -- sample mean becomes the better estimater of the population mean
# This is one of the key ideas of the Central Limit Theorem (CLT)
