import gdown

url = 'https://drive.google.com/uc?id=1hIRNz5-op9WER1VgOolx9lQD26Z82925'
output = 'out/dataset.tsv'
gdown.download(url, output, quiet=False)
