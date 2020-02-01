Machine Learning Project Template

# Introduction
This repository is mlflow and luigi based machine learning project template.

It will make easy to start machine learning project regardless of dataset and framework with data preprocessing.

You can rerun at main task or intermediate preprocessing, such as image cropping.

# Usage
Run following commands for install and training mnist classifier with fully connected network.
```sh
poetry install
PYTHONPATH='.' poetry run luigi --module projects.run_once RunOnceProject --runner image_recognition_trainer --model fcnn --dataset mnistraw --param-path params.yaml --local-scheduler
```

After the running, result files are created under the mlflow directory.

You can see the results by `mlflow ui` and the running tasks by luigi server.

Please see [MLflow Document](https://www.mlflow.org/docs/latest/index.html) and [Luigi Document](https://luigi.readthedocs.io/en/stable/#) for detail usage.

## running configuration
Following parameters should be set for running.
- `module`: Only 'projects.run_once' can be set now.
- `Task Name`: Only 'RunOnceProject' can be set after module name.
- `runner`: Running target. File name under the 'runner' directory can be set.
- `model`: Running model name. Any model allowed by runner can be set.
- `dataset`: Running dataset name. Any dataset allowed by runner can be set.
- `param-path`: Model and dataset and preprocesssing parameters.

## Learning parameters configuration
Parameters configuration can set `param-path` to yaml format file.

It includes model and dataset and preprocesssing section.

### Model configuration

Model configuration is diffrent allowed parameters for model.

Model object extends [KerasClassifierBase](/model/base.py#L79) can be set number of epoch, learning rate, etc.

```yaml
model:
  epochs: 100
  lr: 0.001
```

It means that model run 100 epochs and 0.001 learning rate.

### Dataset configuration

Dataset configuration is diffrent allowed parameters for dataset.

Dataset object extends [ImageClassifierDatasetBase](/dataset/base.py#L58) can be set batch size, etc.

```yaml
dataset:
  batch_size: 128
```

It means that dataset is used with 128 batch.

### Preprocess configuration

Preprocess configuration consits of preprocess projects, these parameters and rerunning target.

`projects` is preprocess projects setting as dictionary. project name and function name pairs are set, projects are ran from top to bottom and use previous project results.

`parameters` is all project configuration. It is used to every preprocess project.

`update_task` is rerun target project name. When `update_task` is set, the running start target project. In contrast, when `update_task` is empty, the running only act runner project.

In any case, when dependent projects have not been acted with configuration parameters, these projects are running.

```yaml
preprocess:
  projects:
    Download:
      dataset.mnist_from_raw.download_data
    Decompose:
      dataset.mnist_from_raw.decompose_data

  parameters:
    name: sample

  update_task: 'Decompose'
```

It means that download and decompose projects are running with name parameter, but if download have not been acted.

## Workflow for your own task
This framework can apply to specific model, specific dataset and specific task.

For example, when you use to image classification task with your custom model,you implement model class extends [KerasClassifierBase](/model/base.py#L79) class.

This class is implemented to learn image classification with `model` property, 
which constructed by Keras functional API.

# Test and linting
Run following command for testing and linting.
```sh
poetry run tox
```

# Todo
- [] hyper parameter search template

# License
MIT License
