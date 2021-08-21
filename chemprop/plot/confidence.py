import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from chemprop.utils.metrics import Gaussian_NLL
from scipy.stats import norm
from scipy.stats import spearmanr


def multiple_evaluation(y_true, y_pred, unc_dict, evaluation_func, save_path):
    results_df = {}
    for i, method_name in enumerate(unc_dict):
        o1, _ = evaluation_func(y_true, y_pred, unc_dict[method_name], method_name=method_name)
        results_df.update(o1)

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(save_path)


def ranking_evaluation(y_true, y_pred, var_estimated, method_name="NoName"):
    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        raise TypeError("The data you entered is wrong.")

    data = np.vstack((y_true, y_pred, var_estimated)).T
    data = data[data[:, 2].argsort()[::-1]]
    y_ordered = data[:, 0:2]

    MAE        =  np.array([mae(y_ordered[i:, 0], y_ordered[i:, 1]) for i in range(len(y_ordered))])
    ERROR      =  np.sort(np.abs(y_ordered[:, 0] - y_ordered[:, 1]))[::-1]
    MAE_ORACLE =  np.array([np.nanmean(ERROR[i:]) for i in range(len(ERROR))])

    # compute AUCO
    AUC_OF_MAE = MAE.sum() / len(MAE)
    AUC_OF_ORACLE = MAE_ORACLE.sum() / len(MAE)
    AUCO = AUC_OF_MAE - AUC_OF_ORACLE

    # compute Error Drop
    ED = MAE[0] / MAE[-1]

    # compute Decreasing Coefficient
    count = 0
    for i in range(len(MAE) - 1):
        if MAE[i + 1] >= MAE[i]:
            count += 1
        else:
            continue
    DC = count / (len(MAE) - 1)

    SPR = spearmanr(np.abs(y_true - y_pred), var_estimated)[0]
    NLL = Gaussian_NLL(y_true, y_pred, var_estimated)

    return {method_name: {'AUCO': AUCO, 'ED': ED, 'DC': DC, 'SPR': SPR, 'NLL': NLL}}, (MAE, MAE_ORACLE)


def cbc_evaluation(y_true,
                   y_pred,
                   var_estimated,
                   method_name="NoName"):
    if np.sum(var_estimated < 0) > 0:
        return {method_name: {'AUCE': None, 'ECE': None, 'MCE': None}}, None

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        raise TypeError("The data you entered is wrong.")

    std_mean = var_estimated ** 0.5
    data = np.vstack((y_true, y_pred, std_mean)).T

    PERCENT = []
    for p in range(100):
        p = p + 1
        count = 0
        for i in range(len(data)):
            upper = norm.ppf((0.5 + p / 200), data[i, 1], data[i, 2])
            if (2 * data[i, 1] - upper) <= data[i, 0] <= upper:
                count += 1
            else:
                continue
        percent = count / len(data)
        PERCENT.append(percent)

    PERCENT = np.array(PERCENT)
    P = np.linspace(0.01, 1.00, 100)

    AUCE = np.sum(abs(PERCENT - P))
    ECE = np.nanmean(abs(PERCENT - P))
    MCE = np.max(abs(PERCENT - P))

    return {method_name: {'AUCE': AUCE, 'ECE': ECE, 'MCE': MCE}}, PERCENT


def CI_evaluation(y_true,
                  y_pred,
                  var_estimated,
                  p=0.9,
                  method_name="NoName"):

    if np.sum(var_estimated < 0) > 0:
        return {method_name: {'RELIABILITY': None, 'EFFICIENCY': None}}, None

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        raise TypeError("The data you entered is wrong.")

    alpha = (1-p)/2
    std_mean  = var_estimated ** 0.5
    y_pred_up  = norm.ppf(1-alpha, y_pred, std_mean)
    y_pred_low = norm.ppf(alpha,   y_pred, std_mean)
    coverage = sum((y_true < y_pred_up) * (y_true > y_pred_low)) / len(y_true)
    return {method_name: {'RELIABILITY': coverage, 'EFFICIENCY': np.mean(np.abs(y_pred_up-y_pred_low))}}, None


def ebc_evaluation(y_true,
                   y_pred,
                   var_estimated,
                   method_name="NoName"):
    if np.sum(var_estimated < 0) > 0:
        return {method_name: {'ENCE': None}}, None
    """value-based method : error-based calibration
       and indices of the method : Expected normalized calibration error"""

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        print("The data you entered is wrong.")

    data = np.vstack((y_true, y_pred, var_estimated)).T
    data = data[data[:, 2].argsort()]  # Sort the data according to the var_estimated
    # data = data[::-1]
    # var_estimated_ordered = data[:,2].T.squeeze()
    K = len(data) // 20  # the number of bins
    bins = np.array_split(data, K, axis=0)

    RMSE = []
    ERROR = []
    for i in range(len(bins)):
        rmse = np.sqrt(mse(bins[i][:, 0], bins[i][:, 1]))
        error = np.sqrt(np.nanmean(bins[i][:, 2]))
        RMSE.append(rmse)
        ERROR.append(error)
    RMSE = np.array(RMSE)
    ERROR = np.array(ERROR)

    # compute expected normalized calibration error
    ENCE = np.nanmean((abs(RMSE - ERROR)) / ERROR)
    return {method_name: {'ENCE': ENCE}}, (RMSE, ERROR)
