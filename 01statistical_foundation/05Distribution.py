from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "process_engineering_data_lyst1750493214144.csv"

data = pd.read_csv(data_path)

# print(data.head())

# TODO: calculate the range of the temperature col
temperature_range = data["temperature"].max() - data["temperature"].min()
# print(f"Range is: {temperature_range}")


# TODO: Calculate IQR of the temp col
q1 = data["temperature"].quantile(0.25)
q3 = data["temperature"].quantile(0.75)

temperature_IQR = q3 - q1
# print(f"IQR value: {temperature_IQR}")

#TODO: Calculate the variance of the pressure col
pressure_variance = data["pressure"].var()
# print(f"Variance: {pressure_variance}")

# TODO: Calculate the Standard deviation of the pressure col.
standard_deviation_pressure = data["pressure"].std()
# print(f"Standard Deviation: {standard_deviation_pressure}")

