"""
Data creation of nominal column named colors
"""

import pandas as pd
import numpy as np

np.random.seed(0)

number_of_sample = 1000

colors = ["Red", "Green", "Yellow", "Crimson", "Purple"]

nominal_data = np.random.choice(colors, number_of_sample)
# print(nominal_data)

# TODO: convert this array dataset into DataFrame
dataframe = pd.DataFrame(nominal_data, columns=["Colors"])
# print(dataframe)


"""
Data creation of the ordinal col named `satisfaction_levels`
"""

satisfaction_levels = ["Low", "Medium", "High"]
ordinal_data = np.random.choice(satisfaction_levels, size=number_of_sample)

# Adding new col to existing dataframe
dataframe["satisfaction_levels"] = ordinal_data
print(dataframe.head())
