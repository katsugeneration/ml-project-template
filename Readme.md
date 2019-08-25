Machine Learning Project Template

# Introduction
This repository is mlflow based machine learning project template.
It will make easy to start machine learning project regardless of dataset and framework.

# Usage
Run following commands for install and training mnist classifier with fully connected network.
```sh
poetry install
poetry run mlflow run . --no-conda -P runner=image_recognition_trainer -P model=fcnn -P dataset=mnist
```

Please see [MLflow Document](https://www.mlflow.org/docs/latest/index.html) for MLflow usage.

## Learning parameters configuration
Parameters configuration can set to `model_param_path` and `dataset_param_path` for mlflow project arguments.

```sh
poetry run mlflow run . --no-conda -P runner=image_recognition_trainer -P model_param_path=model.conf.yaml -P dataset_param_path=dataset.conf.yaml
```

Configuration files are yaml format.
Default model_param_path `model.conf.yaml` has following content.

```yaml:model.conf.yaml
hidden_nums: 512
dropout_rate: 0.2
```

It means that fully connected network has 512 hidden layer units and 0.2 dropout.
So this configuration is changed, you can apply modifiable parameters to model.
Dataset parameters configuration is same as model parameters.

## Workflow for your own task
This framework can apply to specific model, specific dataset and specific task.
For example, when you use to image classification task with your custom model,
you implement model class extends [KerasImageClassifierBase](/model/base.py) class.

This class is implemented to learn image classification with `model` property, 
which constructed by Keras functional API.

Let's See below for details.

- [Model settings and implementation](/model/Readme.md)
- [Dataset settings and implementation](/dataset/Readme.md)
- [Runner implementation](/runner/Readme.md)

# Todo
- [] multiple task chain template
- [] hyper parameter search template
- [] test setting template
- [] CI settings(mypy, flake8, pydocstyle etc...) template

# License
MIT License
