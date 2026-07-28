"""Classification Accuracy"""
from sklearn.metrics import accuracy_score

YTrue = ["Dog", "Dog", "Cat", "Dog", "Cat", "Cat", "Cat", "Dog", "Dog", "Cat"]
YPred = ["Dog", "Cat", "Cat", "Dog", "Dog", "Dog", "Cat", "Dog", "Dog", "Cat"]

# print(accuracy_score(YTrue, YPred) * 100)  # 70.0

"""Confusion Metrics"""
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(YTrue, YPred, labels=["Dog", "Cat"])
# print(confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# ax = sns.heatmap(confusion_matrix, annot=True, cmap="Blues")

# ax.set_title("Confusion matrix with labels\n\n")
# ax.set_xlabel("Predicted Values\n")
# ax.set_ylabel("Actual Values\n")

# Ticket label: List must be in alphabetical order
# ax.xaxis.set_ticklabels(["Dog", "Cat"])
# ax.yaxis.set_ticklabels(["Dog", "Cat"])

# plt.savefig("Confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.show()


"""TODO: Covid vs Not-Covid"""
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

YTrue = ['COVID', 'COVID',     'Not-COVID', 'COVID', 'Not-COVID', 'Not-COVID', 'Not-COVID', 'COVID', 'COVID', 'Not-COVID']

YPred = ['COVID', 'Not-COVID', 'COVID',     'COVID', 'Not-COVID', 'COVID',      'Not-COVID', 'COVID', 'COVID', 'Not-COVID']


cf_matrix = confusion_matrix(YTrue, YPred, labels=["Not-COVID", "COVID"])
# print(cf_matrix)

TN, FP, FN, TP = confusion_matrix(YTrue, YPred).ravel()
# print(TN, FP, FN, TP)

cf_report = classification_report(YTrue, YPred)

ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues")

ax.set_title("Not-Covid and Covid Confusion Matrix")
ax.set_ylabel("Actual values")
ax.set_xlabel("\nPredicted values")

ax.xaxis.set_ticklabels(["Not-COVID", "COVID"])
ax.yaxis.set_ticklabels(["Not-COVID", "COVID"])

print(cf_report)

plt.savefig("Confusion_matrix_covid_and_not-covid.png", dpi=300, bbox_inches="tight")
plt.show()
