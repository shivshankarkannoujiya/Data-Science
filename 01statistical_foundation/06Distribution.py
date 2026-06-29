import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

"""
loc: mean
scale: standard deviation
"""
# Standard normal distribution
"""
Standardised the value in the Normal Distribution
Value will be in a range
"""
portfolio_returns = np.random.normal(loc=0, scale=1, size=1000)


# Adding the extreme value
portfolio_returns = np.random.normal(loc=0, scale=50, size=50)


# TODO: Add a few samples in such a way that it will become positive kurtosis

# TODO: calculate the Kurtosis
from scipy.stats import kurtosis

kurt_value = kurtosis(portfolio_returns)

print(f"Kurtosis: {kurt_value:.2f}")

sns.histplot(portfolio_returns, bins=30, kde=True)
plt.title(f"Portfolio returns distribution: {kurt_value:.2f}")
plt.xlabel("Portfolio returns")
plt.ylabel("Frequency")

# plt.savefig("portfolio_returns.png")
plt.savefig("portfolio_returns_platy.png")

plt.show()
