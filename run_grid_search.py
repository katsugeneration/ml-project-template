# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import itertools
import argparse
import mlflow
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runner', type=str,
                        help="Runner name.")
    parser.add_argument('--model', default='fcnn', type=str,
                        help="Training model name.")
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="Training dataset name.")
    parser.add_argument('--params_path', default="params.conf.yaml", type=str,
                        help="Parameters config file path.")
    args = parser.parse_args()

    # parameter preprocessing
    with open(args.params_path, 'r') as f:
        parameters = yaml.load(f)

    model_param_path = 'model.search.yaml'
    dataset_param_path = 'dataset.search.yaml'

    if 'model' in parameters:
        model_parameters = parameters['model']
    else:
        model_parameters = {}

    if 'dataset' in parameters:
        dataset_parameters = parameters['dataset']
    else:
        dataset_parameters = {}

    with mlflow.start_run() as active_run:
        for mp in itertools.product(*model_parameters.values()):
            with open(model_param_path, 'w') as w:
                yaml.dump({k: v for k, v in zip(model_parameters.keys(), mp)}, w)
            for dp in itertools.product(*dataset_parameters.values()):
                with open(dataset_param_path, 'w') as w:
                    yaml.dump({k: v for k, v in zip(dataset_parameters.keys(), dp)}, w)
                parameters = {
                    "runner": args.runner,
                    "model": args.model,
                    "dataset": args.dataset,
                    "model_param_path": model_param_path,
                    "dataset_param_path": dataset_param_path
                }
                submitted_run = mlflow.run(".", 'main', parameters=parameters, use_conda=False)
