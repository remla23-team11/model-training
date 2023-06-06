from src.helper import get_csv_data
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import joblib

nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def preprocess(dataset):
    corpus = []
    for review in dataset['Review']:
        corpus.append(clean_review(review))

    cv = CountVectorizer(max_features=1420)
    preprocessed_data = cv.fit_transform(corpus).toarray()

    pickle.dump(cv, open('out/c1_BoW_Sentiment_Model.pkl', "wb"))
    joblib.dump(preprocessed_data, 'out/preprocessed.joblib')

    return corpus


def clean_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


if __name__ == '__main__':
    dataset = get_csv_data('a1_RestaurantReviews_HistoricDump.tsv')
    preprocess(dataset)
