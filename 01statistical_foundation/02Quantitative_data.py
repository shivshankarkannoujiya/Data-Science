import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

num_samples = 1000

colors = ["Red", "Green", "Yellow", "Crimson", "Purple"]

nominal_data = np.random.choice(colors, num_samples)
dataframe = pd.DataFrame(nominal_data, columns=["Colors"])


np.random.seed(0)

satisfaction_levels = ["Low", "Medium", "High"]
ordinal_data = np.random.choice(satisfaction_levels, size=num_samples)
dataframe["satisfaction_levels"] = ordinal_data


"""
Data creation of the interval data under Temperature Column
"""

interval_data = np.random.randint(0, 100, num_samples)
dataframe["temperature"] = interval_data


"""
Data creation of the Ratio  data under income Column
"""

np.random.seed(0)

ratio_data = np.random.randint(10000, 1000000, size=num_samples)
dataframe["income"] = ratio_data

# print(dataframe.head())

# Saving file into csv
dataframe.to_csv("types_of_data.csv")


# Create plot of the ratio data
plt.hist(dataframe["income"], bins=20, color="green", edgecolor="black")
plt.title("Income distribution")
plt.xlabel("Income (Rupees)")
plt.ylabel("Frequency")
plt.savefig("income_distribution.png")
plt.show()
