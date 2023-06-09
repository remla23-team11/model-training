from sklearn.metrics import confusion_matrix, accuracy_score
from src.helper import get_csv_data
from src.preprocess import preprocess
from src.model import train, evaluate
from src.load_data import load_data


def main():
    """
    Main function to run the model training pipeline
    """
    load_data()
    dataset = get_csv_data('out/dataset.tsv')
    preprocess(dataset)
    classifier, x_test, y_test = train(dataset)
    y_pred = evaluate(classifier, x_test)

    confusion_m = confusion_matrix(y_test, y_pred)
    print(confusion_m)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
