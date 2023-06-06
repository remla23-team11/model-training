from sklearn.metrics import confusion_matrix, accuracy_score
from src.helper import get_csv_data
from src.preprocess import preprocess
from src.model import train, evaluate


def main():
    dataset = get_csv_data('a1_RestaurantReviews_HistoricDump.tsv')
    preprocess(dataset)
    classifier, X_test, y_test = train(dataset)
    y_pred = evaluate(classifier, X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
