"""Scripts for evaluation metrics"""

import math
from sklearn import metrics
import numpy as np


def calc_score(y_pred, y_true, climatology):
    """
    Compute ACC, TSS, BSS, and GMGS
    """
    score = {}
    y_predl = []
    for y in y_pred:
        y_predl.append(np.argmax(y))

    score["ACC"] = calc_acc4(y_predl, y_true)
    score["TSS-M"] = calc_tss(y_predl, y_true, 2)
    score["BSS-M"] = calc_bss(y_pred, y_true, climatology)
    score["GMGS"] = calc_gmgs(y_predl, y_true)

    return score


def calc_tss(y_predl, y_true, flare_class):
    """
    Compute TSS
    """
    tn, fp, fn, tp = calc_tp_4(
        metrics.confusion_matrix(y_true, y_predl,
                                 labels=[0, 1, 2, 3]), flare_class)
    tss = (tp / (tp+fn)) - (fp / (fp + tn))
    if math.isnan(tss):
        return 0
    return float(tss)


def calc_gmgs(y_predl, y_true):
    """
    Compute GMGS (simplified version)
    """
    tss_x = calc_tss(y_predl, y_true, 3)
    tss_m = calc_tss(y_predl, y_true, 2)
    tss_c = calc_tss(y_predl, y_true, 1)
    print("x: ", tss_x)
    print("m: ", tss_m)
    print("c: ", tss_c)
    return (tss_x + tss_m + tss_c) / 3


def calc_bss(y_pred, y_true, climatology):
    """
    Compute BSS >= M
    """
    y_truel = []
    for y in y_true:
        y_truel.append(convert_2_one_hot_2class(y))

    y_pred2 = np.reshape(np.array(y_pred).ravel(), (-1, 2))
    y_pred2 = np.reshape(np.sum(y_pred2, axis=1), (-1, 2))
    bs = 0
    bsc = 0
    for p, t in zip(y_pred2.tolist(),
                    np.array(y_truel).tolist()):
        bs += (p[1] - t[1]) ** 2
    bs = bs / len(y_true)
    bsc = climatology[0] * climatology[1]
    bss = (bsc - bs) / bsc

    return bss


def convert_2_one_hot_2class(binary_value):
    """
    return 2-dimentional 1-of-K vector
    """
    if binary_value == 3 or binary_value == 2:
        return [0, 1]
    return [1, 0]


def calc_cm4(y_pred, y_true):
    """
    return confusion matrix for 4 class
    """
    return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])


def calc_acc4(y_pred, y_true):
    """
    Compute classification accuracy for 4 class
    """
    cm = calc_cm4(y_pred, y_true)
    print(cm)
    acc4 = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / len(y_pred)
    return acc4


def calc_tp_4(four_class_matrix, flare_class):
    """
    Convert 4 class output to 2 class
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for true in range(4):
        for pred in range(4):
            if true >= flare_class:
                if pred >= flare_class:
                    tp += four_class_matrix[true][pred]
                else:
                    fn += four_class_matrix[true][pred]
            else:
                if pred >= flare_class:
                    fp += four_class_matrix[true][pred]
                else:
                    tn += four_class_matrix[true][pred]
    return tn, fp, fn, tp
