import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
- 5 Number Summary and Box Plot
- Minimum, Maximum, Median, Q1, Q2, Q3, IQR
"""

marks = [45, 32, 56, 75, 89, 54, 32, 89, 90, 87, 67, 54, 45, 98, 99, 67, 74]
minimum, Q1, median, Q3, maximum = np.quantile(marks, [0, 0.25, 0.50, 0.75, 1.0])

# print(minimum, Q1, median, Q3, maximum)

IQR = Q3 - Q1

lower_fense = Q1 - 1.5 * (IQR)
higer_fense = Q3 + 1.5 * (IQR)

# print(f"lower_fense: {lower_fense}, higher_fense: {higer_fense}")

# sns.boxplot(marks)
# plt.show()


marks_with_outliers = [
    -100,
    -200,
    45,
    32,
    56,
    75,
    89,
    54,
    32,
    89,
    90,
    87,
    67,
    54,
    45,
    98,
    99,
    67,
    74,
    150,
    170,
    180,
]


sns.boxplot(marks_with_outliers)
plt.show()
