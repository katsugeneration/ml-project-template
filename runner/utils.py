# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Union, List, Tuple
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


def save_confusion_matrix(
        pred: np.array,
        label: np.array,
        num_classes: int,
        path: Union[pathlib.Path, str]) -> None:
    """Save confution matrix to target path.

    Args:
        pred (np.array): prediction array.
        label (np.array): truth label array.
        num_classes (int): number of classes.
        path (Union[pathlib.Path, str]): path to save image file.

    """
    labels = list(range(num_classes))
    cm = confusion_matrix(label, pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(12.8, 7.2))
    seaborn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True)
    fig.savefig(str(path))


def compute_class_accuracies(
        pred: np.array,
        label: np.array,
        num_classes: int) -> List[float]:
    """Compute accuracy per classes.

    Args:
        pred (np.array): prediction array.
        label (np.array): truth label array.
        num_classes (int): number of classes.

    Return:
        accuracies (List[float]): accuracy per classes.

    """
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 0.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(count[i] / total[i])
    return accuracies


def compute_mean_iou(
        pred: np.array,
        label: np.array) -> float:
    """Compute mean IoU.

    Args:
        pred (np.array): prediction array.
        label (np.array): truth label array.

    Return:
        mean_iou (float): mean IoU.

    """
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(
        pred: np.array,
        label: np.array,
        num_classes: int,
        score_averaging: str = "weighted") -> Tuple[List[float], float, List[float], float, float, float]:
    """Compute image segmentation evaluation metrics.

    Args:
        pred (np.array): prediction array.
        label (np.array): truth label array.
        num_classes (int): number of classes.
        score_averaging (str): score averaging type.

    Return:
        class_accuracies (List[float]): accuracy per classes.
        prec (float): mean precision.
        class_prec (List[float]): precision per classes.
        rec (float): mean recall.
        f1 (float): mean f1 score.
        iou (float): mean IoU.

    """
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)
    prec = precision_score(flat_label, flat_pred, average=score_averaging)
    class_prec = precision_score(flat_label, flat_pred, average=None, labels=list(range(num_classes)))
    rec = recall_score(flat_label, flat_pred, average=score_averaging)
    f1 = f1_score(flat_label, flat_pred, average=score_averaging)
    iou = compute_mean_iou(flat_pred, flat_label)

    return class_accuracies, prec, class_prec, rec, f1, iou
