import random
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score
from mutatest.mutators import ReplacementMutator
from src.preprocess import preprocess
from src.helper import get_csv_data
from src.model import train, evaluate
from src import app


@pytest.fixture
def reviews():
    """
    Fixture that provides example review strings as input data.
    """
    example_strings = [
        "This hole in the wall has great Mexican street tacos, and friendly staff",
        """Took an hour to get our food only 4 tables in restaurant my food was Luke warm, Our
		sever was running around like he was totally overwhelmed""",
        "The worst was the salmon sashimi",
        "Also there are combos like a burger, fries, and beer for 23 which is a decent deal",
        "This was like the final blow!",
        "I found this place by accident and I could not be happier",
        """seems like a good quick place to grab a bite of some familiar pub food, but do yourself
		a favor and look elsewhere""",
        "Overall, I like this place a lot",
        "The only redeeming quality of the restaurant was that it was very inexpensive",
        "Ample portions and good prices",
        "Poor service, the waiter made me feel like I was stupid every time he came to the table",
        "My first visit to Hiro was a delight!"
    ]

    return pd.DataFrame({'Review': example_strings})


@pytest.fixture
def actual_reviews():
    """
    Fixture that provides the expected processed reviews.
    """
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
    """
    Fixture that provides the test dataset loaded from a CSV file.
    """
    return get_csv_data('tests/data/RestaurantReviews.tsv')


def test_get_csv_data(reviews_data):
    """
    Test if the dataset is successfully loaded.
    """
    assert len(reviews_data) > 0
    assert reviews_data.shape[1] == 2


def test_preprocess(reviews, actual_reviews):
    """
    Test the preprocess function.
    """
    processed_reviews = preprocess(reviews)
    assert processed_reviews == actual_reviews


def test_data_slice(reviews_data):
    """
    Test data slicing.
    Due to lack of distinctive, unbiased features, we are assuming that
    the reviews are chronological and we are taking the first 400.
    """
    preprocess(reviews_data)
    classifier, x_test, y_test = train(reviews_data)
    y_pred = evaluate(classifier, x_test)
    original_acc = accuracy_score(y_test, y_pred)

    ones = reviews_data[reviews_data['Liked'] == 1].head(200)
    zeros = reviews_data[reviews_data['Liked'] == 0].head(200)
    sliced_data = pd.concat([ones, zeros], ignore_index=True)
    preprocess(sliced_data)
    classifier, x_test, y_test = train(sliced_data)
    y_pred = evaluate(classifier, x_test)
    sliced_acc = accuracy_score(y_test, y_pred)

    assert abs(original_acc - sliced_acc) < 0.1


def test_train(reviews_data):
    """
    Test the training process.
    """
    preprocess(reviews_data)
    accs = []

    for _ in range(5):

        classifier, x_test, y_test = train(
            reviews_data, random.randint(1, 100))
        y_pred = evaluate(classifier, x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accs.append(accuracy)

    assert all(abs(acc1 - acc2) <= 0.2 for i, acc1 in enumerate(accs)
               for acc2 in accs[i+1:])


def test_main():
    """
    Test the integrated main function of the app.
    """
    app.main()


def test_mutamorphic(reviews_data):
    """
    Test mutamorphic mutation.
    """
    preprocess(reviews_data)
    classifier, x_test, y_test = train(reviews_data)
    y_pred = evaluate(classifier, x_test)
    original_acc = accuracy_score(y_test, y_pred)

    mutated_data = reviews_data.copy()
    mutator = ReplacementMutator(num_variants=1)

    for _ in range(5):
        mutated_data = reviews_data.copy()

        mutated_data['Review'] = reviews_data['Review'].apply(
            lambda x: mutator.mutate(x, random_seed=random.randint(1, 100))[0])

        preprocess(mutated_data)
        classifier, _, _ = train(
            mutated_data, random.randint(1, 100))
        y_pred = evaluate(classifier, x_test)
        mutated_acc = accuracy_score(y_test, y_pred)

        assert abs(original_acc - mutated_acc) < 0.3
