Machine Learning Project Template

# Introduction
This repository is mlflow based machine learning project template.
It will make easy to start machine learning project regardless of dataset and framework.

# Usage
Run under the commands for install and training mnist classifier with fully connected network.
```sh
poetry install
poetry run mlflow run . --no-conda -P runner=image_recognition_trainer -P model=fcnn -P dataset=mnist
```

## Use another model


## Use another dataset


## Use another task


# Workflow for your own task
1. dataset class implementation
2. task base class implementation depend on task dataset
3. specific model implementation
4. task train runner implementation

# Todo
- [] multiple train path template
- [] hyper parameter search template
- [] test setting template
- [] CI settings(mypy, flake8, pydocstyle etc...) template
