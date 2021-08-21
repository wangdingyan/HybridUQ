import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chemprop.conformal.evaluation import *
# https://github.com/msesia/cqr-comparison/blob/0a3fa763e6b35e01290f10c53797367112731a75/third_party/cqr/helper.py


def conformal_curves(test_labels,
                     test_conformal_dict):
    results_dict = {}
    for model in test_conformal_dict:
        rel = reliability(test_labels,
                          test_conformal_dict[model]['L'],
                          test_conformal_dict[model]['U'])
        eff = np.abs(test_conformal_dict[model]['U']-test_conformal_dict[model]['L']).mean()
        results_dict.update({model:{'Reliability': rel, 'Efficiency': eff}})
        results_df = pd.DataFrame(results_dict)
    return results_df


def plot_func_data(y_test,
                   y_lower,
                   y_upper,
                   name=""):
    """ Plot the test labels along with the constructed prediction band
    Parameters
    ----------
    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    name : string, optional output string (e.g. the method name)
    """

    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = y_upper - y_lower
    sort_ind = np.argsort(interval)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]
    mean = (upper_sorted + lower_sorted) / 2

    # Center such that the mean of the prediction interval is at 0.0
    y_test_sorted -= mean
    upper_sorted -= mean
    lower_sorted -= mean

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

    interval = y_upper - y_lower
    sort_ind = np.argsort(y_test)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples by response")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()