# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import shutil
import pathlib
import argparse
import importlib
import yaml
import mlflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runner', type=str,
                        help="Runner name.")
    parser.add_argument('--model', default='fcnn', type=str,
                        help="Training model name.")
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="Training dataset name.")
    parser.add_argument('--model_param_path', default="model.conf.yaml", type=str,
                        help="Model config file path.")
    parser.add_argument('--dataset_param_path', default="dataset.conf.yaml", type=str,
                        help="Dataset config file path.")
    parser.add_argument('--logs', default='logs',
                        type=str, help="Path to output log directory.")
    args = parser.parse_args()

    # parameter preprocessing
    with open(args.model_param_path, 'r') as f:
        model_params = yaml.load(f)
    mlflow.log_params(model_params)

    with open(args.dataset_param_path, 'r') as f:
        dataset_params = yaml.load(f)
    mlflow.log_params(dataset_params)

    logs = pathlib.Path(args.logs)
    if logs.exists():
        shutil.rmtree(str(logs))
    logs.mkdir(parents=True)

    # do runner
    module = importlib.import_module('runner.' + args.runner)
    class_name = "".join(s[:1].upper() + s[1:] for s in args.runner.split('_'))
    c = getattr(module, class_name)
    history = c().run(args.model, args.dataset, model_params, dataset_params, logs)

    # save to mlflow
    for k in history:
        for i in range(len(history[k])):
            mlflow.log_metric(k, history[k][i], step=i)
    mlflow.log_artifacts(str(logs))
