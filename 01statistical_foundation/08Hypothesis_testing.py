from scipy.stats import norm

population_mean = 800
sample_mean = 790
standard_error = 7.5
alpha = 0.05

z_score = (sample_mean - population_mean) / standard_error

# Two tailed value
p_value = 2 * norm.cdf(z_score)

if p_value > alpha:
    print("Fail to reject Null Hypothesis")
else:
    print("Reject Null Hypothesis")
