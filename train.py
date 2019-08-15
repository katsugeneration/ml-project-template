import pathlib
import yaml
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from model.fcnn import FCNNClassifier
from model.resnet import ResNet
from dataset.mnist import MnistDataset
from dataset.cifar10 import Cifar10Dataset


def main(args):
    # prepare dataset
    with open(args.dataset_param_path, 'r') as f:
        dataset_params = yaml.load(f)
    if args.dataset == 'mnist':
        dataset = MnistDataset(**dataset_params)
    elif args.dataset == 'cifar10':
        dataset = Cifar10Dataset(**dataset_params)
    else:
        raise NameError("dataset {} not found" % (args.dataset))

    # prepare model
    with open(args.model_param_path, 'r') as f:
        model_params = yaml.load(f)
    model_params['dataset'] = dataset
    if args.model == 'fcnn':
        classifier = FCNNClassifier(**model_params)
    elif args.model == 'resnet':
        classifier = ResNet(**model_params)
    else:
        raise NameError("model {} not found" % (args.model))

    # run learning
    history = classifier.train()
    classifier.save(pathlib.Path(args.logs).joinpath('model.h5'))

    # save results
    y_pred, y_test = classifier.predict()
    y_pred = np.argmax(y_pred, axis=1)

    labels = list(range(10))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(12.8, 7.2))
    seaborn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True)
    fig.savefig(str(logs.joinpath('confusion_matrix.png')))

    return history


if __name__ == '__main__':
    import argparse
    import mlflow
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="Training dataset name.")
    parser.add_argument('--model', default='fcnn', type=str,
                        help="Training model name.")
    parser.add_argument('--dataset_param_path', default="model.conf.yaml", type=str,
                        help="Number of epoch.")
    parser.add_argument('--model_param_path', default="model.conf.yaml", type=str,
                        help="Number of epoch.")
    parser.add_argument('--logs', default='logs',
                        type=str, help="Path to output log directory.")
    args = parser.parse_args()

    with open(args.dataset_param_path, 'r') as f:
        dataset_params = yaml.load(f)
    mlflow.log_params(dataset_params)

    with open(args.model_param_path, 'r') as f:
        model_params = yaml.load(f)
    mlflow.log_params(model_params)

    logs = pathlib.Path(args.logs)
    if not logs.exists():
        logs.mkdir(parents=True)

    history = main(args)

    # save to mlflow
    for k in history:
        for i in range(len(history[k])):
            mlflow.log_metric(k, history[k][i], step=i)
    mlflow.log_artifacts(str(logs))
