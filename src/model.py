from sklearn.metrics import precision_score, auc, roc_curve, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import os
import json
from src.helper import get_csv_data


def train_model(x, y):
    classifier = GaussianNB()
    classifier.fit(x, y)

    joblib.dump(classifier, 'out/c2_Classifier_Sentiment_model.joblib')

    return classifier


def set_scores(res, pred):

    fpr, tpr, _ = roc_curve(res, pred)
    auc_val = auc(fpr, tpr)
    ac = accuracy_score(res, pred)
    pc = precision_score(res, pred)

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


def train(dataset, seed=42):
    X = joblib.load('out/preprocessed.joblib')
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed)

    classifier = train_model(X_train, y_train)

    return classifier, X_test, y_test


def evaluate(classifier, x):
    return classifier.predict(x)


if __name__ == '__main__':
    # maybe give option to load test preprocessed data
    dataset = get_csv_data('a1_RestaurantReviews_HistoricDump.tsv')
    classifier, X_test, y_test = train(dataset)
    y_pred = evaluate(classifier, X_test)
    set_scores(y_test, y_pred)
