import numpy as np
import pandas as pd
import pickle
import joblib



cv = CountVectorizer(max_features=1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

bow_path = '../out/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

joblib.dump(classifier, '../c2_Classifier_Sentiment_Model')

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)
