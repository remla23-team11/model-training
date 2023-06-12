import os
import json
import joblib
from sklearn.metrics import precision_score, auc, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from src.helper import get_csv_data


def train_model(x_data, y_data):
    """
    Trains the model and saves it as a joblib file.

    Args:
        x_data (numpy.ndarray): Input features.
        y_data (numpy.ndarray): Target labels.

    Returns:
        sklearn.naive_bayes.GaussianNB: Trained classifie
    """
    classifier = GaussianNB()
    classifier.fit(x_data, y_data)

    joblib.dump(classifier, 'out/c2_Classifier_Sentiment_model.joblib')

    return classifier


def set_scores(res, pred):
    """
    Calculates and sets the scores (AUC, accuracy, precision) and saves them to a JSON file.

    Args:
        res (numpy.ndarray): True labels.
        pred (numpy.ndarray): Predicted labels.
    """

    fpr, tpr, _ = roc_curve(res, pred)
    auc_val = auc(fpr, tpr)
    accuracy = accuracy_score(res, pred)
    precision = precision_score(res, pred)

    summary_file = os.path.join("summary.json")
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": accuracy,
                "AUC": auc_val,
                "precision": precision
            },
            file
        )


def train(dataset, seed=42):
    """
    Trains the classifier using the given dataset.

    Args:
        dataset (pandas.DataFrame): Input dataset.
        seed (int): Random seed for train-test splitting. Defaults to 42.

    Returns:
        tuple: Trained classifier, x_test, y_test.
    """
    x_data = joblib.load('out/preprocessed.joblib')
    y_data = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=seed)

    classifier = train_model(x_train, y_train)

    return classifier, x_test, y_test


def evaluate(classifier, x_data):
    """
    Evaluates the classifier by predicting the labels for the given input.

    Args:
        classifier (sklearn.naive_bayes.GaussianNB): Trained classifier.
        x (numpy.ndarray): Input features.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    return classifier.predict(x_data)


if __name__ == '__main__':
    # maybe give option to load test preprocessed data
    dataset_main = get_csv_data('a1_RestaurantReviews_HistoricDump.tsv')
    cls, x_test_main, y_test_main = train(dataset_main)
    y_pred = evaluate(cls, x_test_main)
    set_scores(y_test_main, y_pred)
