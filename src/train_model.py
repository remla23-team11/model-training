import json
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)

X = joblib.load('out/preprocessed.joblib')
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'out/c2_Classifier_Sentiment_model.joblib')

y_pred = classifier.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_val = auc(fpr, tpr)

ac = accuracy_score(y_test, y_pred)

pc = precision_score(y_test, y_pred)

summary_file = os.path.join("summary.json")
with open(summary_file, "w") as fd:
    json.dump(
        {
            "accuracy": ac,
            "AUC": auc_val,
            "precision": pc
        },
        fd
    )
