import pytest

import numpy as np
from src.preprocess import preprocess

reviews = [
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


def test_preprocess():

    processed_reviews = preprocess(reviews)

    print(processed_reviews)
    return "test"
