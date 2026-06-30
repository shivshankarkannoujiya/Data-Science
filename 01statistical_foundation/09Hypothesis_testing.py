from scipy.stats import norm

population_mean = 20
sample_mean = 18.8
standard_error = 0.8
alpha = 0.05

z_score = (sample_mean - population_mean) / standard_error
print(z_score)

p_value = 2 * norm.cdf(z_score)
print(p_value)

if p_value > alpha:
    print("Fail to reject Null Hypothesis")
else:
    print("Reject Null Hypothesis")
