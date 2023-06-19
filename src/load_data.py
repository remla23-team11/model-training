import gdown


def load_data():
    """
        Loads the test dataset from Google Drive and stores it at out/dataset.tsv
    """
    url = 'https://drive.google.com/uc?id=1hIRNz5-op9WER1VgOolx9lQD26Z82925'
    output = 'out/dataset.tsv'
    gdown.download(url, output, quiet=False)


if __name__ == '__main__':
    load_data()
