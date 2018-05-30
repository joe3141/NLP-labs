import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

train = pd.read_csv('train_f.csv').values
test = pd.read_csv('test_f.csv').values

X_train = train[:, :train.shape[1]-3]
y_train = train[:, train.shape[1]-1]

X_test = test[:, :test.shape[1]-3]
y_test = test[:, test.shape[1]-1]

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

mnb_preds = mnb.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_preds = knn.predict(X_test)


rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)


print("Naive Bayes Report:\n")
print(classification_report(y_test, mnb_preds))

print("\n\nKNN Report:\n")
print(classification_report(y_test, knn_preds))

print("\n\nRandom Forest Report:\n")
print(classification_report(y_test, rf_preds))
