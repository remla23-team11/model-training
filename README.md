# Model Training
Repository containing the ML training pipeline

## Installation
Dependencies can be installed using the requirements.txt, the Python version used to run this project is 3.9.16, if you don't have this installed consider using [pyenv](https://github.com/pyenv/pyenv).

Dependencies can be installed using the requirements.txt
```shell
pip install -r requirements.txt
```

## Running the Pipeline using DVC
[DVC (Data Version Control)](https://dvc.org/) is a version control system for data and ML models. It helps you manage your ML experiments and models efficiently. This project uses DVC to manage the ML training pipeline.

This dvc is set up in two different stages, the preprocessing_dataset stage and the train_model stage. 

To fetch the current version of the pipeline one can use: `dvc pull`.

To run the pipeline use `dvc repro` and to force a rune use `dvc repro -f`.

Now you can see the metrics by using `dvc metrics show`.

## Running the Pipeline Normally

If you prefer not to use DVC, you can run the pipeline directly.

```shell
python -m src.app
```

## Testing
This project uses pytest for testing. To run the tests, use the following command:

```shell
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).