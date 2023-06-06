import pytest

import pandas as pd
from src.preprocess import preprocess


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


def test_preprocess(reviews, actual_reviews):

    processed_reviews = preprocess(reviews)
    assert processed_reviews == actual_reviews
