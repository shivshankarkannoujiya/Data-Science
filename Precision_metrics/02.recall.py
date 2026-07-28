from sklearn.metrics import recall_score, classification_report

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

"""
- Calculate metrics for each label, and find their unweighted mean.
- This does not take label imbalance into account.
"""
print(recall_score(y_true, y_pred, average="macro"))


"""
- Calculate metrics globally by counting the total true positives, False Negative and False positives.
"""
print(recall_score(y_true, y_pred, average="micro"))

"""
- Calculate metrics for each lebel, and find their avg weighted by support
- Number of true instances for each label
"""
print(recall_score(y_true, y_pred, average="weighted"))


print(classification_report(y_true, y_pred, target_names=["0", "1", "2"]))
