from sklearn.metrics import precision_score, classification_report

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print(precision_score(y_true, y_pred, average="macro"))
print(precision_score(y_true, y_pred, average="micro"))
print(precision_score(y_true, y_pred, average="weighted"))

print(classification_report(y_true, y_pred, target_names=["0", "1", "2"]))