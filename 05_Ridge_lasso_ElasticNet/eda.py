from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ROOT_DIR = Path(__file__).parent

data_path = ROOT_DIR / "Algerian_forest_fires_dataset.csv"

dataset = pd.read_csv(data_path)

# print(dataset)
# print(dataset.info())

"""Data Cleaning"""

"""1. Check missing values"""
# print(dataset.isnull().sum())
print(dataset[dataset.isnull().any(axis=1)])