import seaborn as sns

df = sns.load_dataset("tips")
# print(df.head())

"""
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
"""

# TODO: Encode time based on the total value
mean_time  = df.groupby("time")["total_bill"].mean().to_dict()
# print(mean_time)  # {'Lunch': 17.168676470588235, 'Dinner': 20.79715909090909}

df["time_encoded"] = df["time"].map(mean_time)

# print(df)

print(df[["total_bill", "time_encoded"]])

"""
0         16.99    20.797159
1         10.34    20.797159
2         21.01    20.797159
3         23.68    20.797159
4         24.59    20.797159
..          ...          ...
239       29.03    20.797159
240       27.18    20.797159
241       22.67    20.797159
242       17.82    20.797159
243       18.78    20.797159
"""