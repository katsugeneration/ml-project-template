import shutil
import pathlib
import yaml
import numpy as np
from PIL import Image
from model.gan import Gan
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
    model = Gan(**model_params)

    # run learning
    history = model.train()
    model.save(pathlib.Path(args.logs).joinpath('model'))

    # save results
    logs = pathlib.Path(args.logs)

    y_pred, y_test = model.inference()
    for i in range(y_pred.shape[0]):
        if y_pred.shape[3] == 1:
            img = Image.fromarray(np.uint8(y_pred[i, :, :, 0]))
        else:
            img = Image.fromarray(np.uint8(y_pred[i]))
        img.save(str(logs.joinpath('predictions_{:04d}.png'.format(i))))
    for i in range(y_test.shape[0]):
        if y_test.shape[3] == 1:
            img = Image.fromarray(np.uint8(y_test[i, :, :, 0]))
        else:
            img = Image.fromarray(np.uint8(y_test[i]))
        img.save(str(logs.joinpath('gt_{:04d}.png'.format(i))))

    return history


if __name__ == '__main__':
    import argparse
    import mlflow
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="Training dataset name.")
    parser.add_argument('--model', default='fcnn', type=str,
                        help="Training model name.")
    parser.add_argument('--dataset_param_path', default="dataset.conf.yaml", type=str,
                        help="Dataset config file path.")
    parser.add_argument('--model_param_path', default="model.conf.yaml", type=str,
                        help="Model config file path.")
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
    if logs.exists():
        shutil.rmtree(str(logs))
    logs.mkdir(parents=True)

    history = main(args)

    # save to mlflow
    for k in history:
        for i in range(len(history[k])):
            mlflow.log_metric(k, history[k][i], step=i)
    mlflow.log_artifacts(str(logs))
