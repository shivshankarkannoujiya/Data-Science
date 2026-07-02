import numpy as np
import scipy.stats as stats

observed_data = np.array([[50, 30, 20], [30, 40, 30], [20, 30, 50]])
alpha = 0.05

# TODO: Perform the Chi-square test

chi_sq_stats, p_value, df, expected = stats.chi2_contingency(observed_data)

# print(chi_sq_stats, p_value, df, expected)

if p_value > alpha:
    print("Fail to reject Null Hypothesis")
else:
    print("Reject Null Hypothesis")