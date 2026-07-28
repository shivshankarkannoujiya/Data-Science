from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

wine = datasets.load_wine()

x = wine.data
y = wine.target

class_names = wine.target_names
# print(class_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

classifier_tree = DecisionTreeClassifier()
y_predict = classifier_tree.fit(x_train, y_train).predict(x_test)

print(classification_report(y_test, y_predict, target_names=class_names))
print(confusion_matrix(y_test, y_predict))
