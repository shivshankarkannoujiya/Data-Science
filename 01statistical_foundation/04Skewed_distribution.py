import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew


# Data creation for the income
np.random.seed(42)

sample_size = 1000

# Right skewed distribution
income_distribution = np.random.gamma(shape=2.0, scale=1000.0, size=sample_size)

# Calculate the skewness
skew_value = skew(income_distribution)
# print(skew_value) 1.1962783386596858

sns.histplot(income_distribution, bins=30, kde=True)
plt.title(f"Income Distribution: {skew_value: .2f} ")
plt.xlabel("Income")
plt.ylabel("Frequency")

# plt.savefig("right_skewed_income_distribution.png", dpi=300, bbox_inches="tight")

plt.show()
