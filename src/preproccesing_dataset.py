import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib

from preprocessing import clean_review

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)

corpus = []
for review in dataset['Review']:
    corpus.append(clean_review(review))

cv = CountVectorizer(max_features=1420)
preprocessed_data = cv.fit_transform(corpus).toarray()

bow_path = 'out/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

joblib.dump(preprocessed_data, 'out/preprocessed.joblib')
