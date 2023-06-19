import re
import pickle
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from src.helper import get_csv_data

nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def preprocess(dataset):
    """
    Preprocesses the dataset by cleaning the reviews and converting
    them to a Bag-of-Words representation.

    Args:
        dataset (pandas.DataFrame): Input dataset.

    Returns:
        list: Preprocessed reviews.
    """
    corpus = []
    for review in dataset['Review']:
        corpus.append(clean_review(review))

    vectorizer = CountVectorizer(max_features=1420)
    preprocessed_data = vectorizer.fit_transform(corpus).toarray()

    with open('out/c1_BoW_Sentiment_Model.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    joblib.dump(preprocessed_data, 'out/preprocessed.joblib')

    return corpus


def clean_review(review):
    """
    Cleans a review by removing non-alphabetic characters, converting to
    lowercase, removing stopwords, and performing stemming.

    Args:
        review (str): Input review.

    Returns:
        str: Cleaned review.
    """
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


if __name__ == '__main__':
    dataset_main = get_csv_data('out/dataset.tsv')
    preprocess(dataset_main)
