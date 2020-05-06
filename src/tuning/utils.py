"""Function for find optimal prediction
"""

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np


def find_optimal_cutoff_auc(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Credit to this post : https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])
