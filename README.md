[![Coverage Status](https://coveralls.io/repos/github/remla23-team11/model-training/badge.svg?branch=main)](https://coveralls.io/github/remla23-team11/model-training?branch=main)

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

This dvc is set up in three different stages, load_data the preprocess stage and the model stage. 

To fetch the current version of the pipeline one can use: `dvc pull`.

To run the pipeline use `dvc repro` and to force a rune use `dvc repro -f`.

You can see the metrics by using `dvc metrics show`.

## Running the Pipeline Normally

If you prefer not to use DVC, you can run the pipeline directly.

```shell
python -m src.app
```

## Code Quality
To verify the code quality there are two libraries that are used, pylint and mllint, you can use them in the following way:

For pylint with the DSLinter plugin use the following command:
```shell
pylint --load-plugins=dslinter src/ --exit-zero
```

For mllint use the following command:
```shell
mllint
```

## Testing
This project uses pytest for testing. To run the tests, use the following command:

```shell
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).