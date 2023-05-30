# model-training
Repository containing the ML training pipeline

Dependencies can be installed using the requirements.txt, the Python version used to run this project is 3.9.16, if you
don't have this installed consider using [pyenv](https://github.com/pyenv/pyenv).

## DVC
This dvc is set up in two different stages, the preprocessing_dataset stage and the train_model stage. 

To fetch the current version of the pipeline one can use: `dvc pull`.

To run the pipeline use `dvc pull` and to force a rune use `dvc pull -f`.

Now you can see the metrics by using `dvc metrics show`.