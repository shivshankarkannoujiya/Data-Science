import scipy.stats as stats

group_A = [105, 110, 98, 107, 103]
group_B = [130, 125, 132, 128, 129]
group_C = [120, 115, 117, 119, 121]

alpha = 0.05

f_value, p_value = stats.f_oneway(group_A, group_B, group_C)

if alpha < p_value:
    print(
        f"Fail to rejct the null Hypothesis, meaning there is no significat different in the average caffience content in the above three energy groups"
    )
else:
    print(
        f"Reject the Null Hypothesis, meaning there is significant diff in the average caffience content in the above three energy groups"
    )
