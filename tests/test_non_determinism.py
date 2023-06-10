import joblib
import pytest
import random
import pandas as pd
from src.preprocess import preprocess
from src.helper import get_csv_data
from src.model import train, evaluate
import src.app as app
from sklearn.metrics import accuracy_score


@pytest.fixture
def reviews():
    example_strings = [
        "This hole in the wall has great Mexican street tacos, and friendly staff",
        "Took an hour to get our food only 4 tables in restaurant my food was Luke warm, Our sever was running around like he was totally overwhelmed",
        "The worst was the salmon sashimi",
        "Also there are combos like a burger, fries, and beer for 23 which is a decent deal",
        "This was like the final blow!",
        "I found this place by accident and I could not be happier",
        "seems like a good quick place to grab a bite of some familiar pub food, but do yourself a favor and look elsewhere",
        "Overall, I like this place a lot",
        "The only redeeming quality of the restaurant was that it was very inexpensive",
        "Ample portions and good prices",
        "Poor service, the waiter made me feel like I was stupid every time he came to the table",
        "My first visit to Hiro was a delight!"
    ]

    return pd.DataFrame({'Review': example_strings})


@pytest.fixture
def actual_reviews():
    return [
        'hole wall great mexican street taco friendli staff',
        'took hour get food tabl restaur food luke warm sever run around like total overwhelm',
        'worst salmon sashimi',
        'also combo like burger fri beer decent deal',
        'like final blow',
        'found place accid could not happier',
        'seem like good quick place grab bite familiar pub food favor look elsewher',
        'overal like place lot',
        'redeem qualiti restaur inexpens',
        'ampl portion good price',
        'poor servic waiter made feel like stupid everi time came tabl',
        'first visit hiro delight'
    ]


@pytest.fixture
def reviews_data():
    return get_csv_data('tests/data/RestaurantReviews.tsv')


def test_get_csv_data(reviews_data):
    # Test if the dataset is successfully loaded
    assert len(reviews_data) > 0
    assert reviews_data.shape[1] == 2


def test_preprocess(reviews, actual_reviews):

    processed_reviews = preprocess(reviews)
    assert processed_reviews == actual_reviews


def test_train(reviews_data):
    preprocess(reviews_data)

    accs = []

    for _ in range(10):

        classifier, X_test, y_test = train(
            reviews_data, random.randint(1, 100))
        y_pred = evaluate(classifier, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accs.append(accuracy)

    assert all(abs(acc1 - acc2) <= 0.2 for i, acc1 in enumerate(accs)
               for acc2 in accs[i+1:])


def test_main():
    "Integration test app"
    app.main()
