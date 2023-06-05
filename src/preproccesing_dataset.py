import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from preprocessing import clean_review

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
reviews = dataset['Review']

corpus = []
for review in reviews:
    corpus.append(clean_review(review))

cv = CountVectorizer(max_features=1420)
preprocessed_data = cv.fit_transform(corpus).toarray()

BOW_PATH = 'out/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(BOW_PATH, "wb"))

joblib.dump(preprocessed_data, 'out/preprocessed.joblib')
