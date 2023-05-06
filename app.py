import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import joblib
from preprocessing import clean_review

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)

corpus = []
for review in dataset['Review']:
    corpus.append(clean_review(review))

cv = CountVectorizer(max_features=1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

bow_path = 'c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)
