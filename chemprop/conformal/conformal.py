import numpy as np
from sklearn.base import BaseEstimator
from chemprop.conformal.nonconformist import *


# class MCP:
#     def __init__(self,
#                  y_calibrate:     np.array,
#                  y_calibrate_hat: np.array,
#                  calibrate_error_estimated: np.array,
#                  p=0.9):
#
#         assert y_calibrate.shape == y_calibrate_hat.shape == calibrate_error_estimated.shape
#         assert len(y_calibrate.shape) == 1
#         self.y_calibrate = y_calibrate
#         self.y_calibrate_hat = y_calibrate_hat
#         self.calibrate_error_estimated = calibrate_error_estimated
#         self.nonconformity_values = np.abs(self.y_calibrate-self.y_calibrate_hat) / self.calibrate_error_estimated
#         self.p = p
#         self.alpha = np.sort(self.nonconformity_values)[int(len(self.nonconformity_values)*self.p)]
#
#     def predict(self,
#                 y_predicted_hat:           np.array,
#                 predicted_error_estimated: np.array):
#         return y_predicted_hat-self.alpha*predicted_error_estimated, \
#                y_predicted_hat+self.alpha*predicted_error_estimated
#
#     def evaluate(self,
#                  y_predicted:              np.array,
#                  y_predicted_hat:          np.array,
#                  predicted_error_estimated:np.array):
#         intervals = self.alpha*predicted_error_estimated
#         absolute_errors = np.abs(y_predicted-y_predicted_hat)
#         reliability = (intervals > absolute_errors).mean()
#         efficiency  = 2*np.mean(intervals)
#         return reliability, efficiency


class RegressorNC():
    def __init__(self,
                 err_func=AbsErrorErrFunc(),
                 normalizer=lambda x: np.exp(x),
                 beta=1e-6):
        self.error_func = err_func
        self.normalizer = normalizer
        self.beta = beta

    def score(self,
              prediction,
              y,
              error_est=None,):
        n_test = prediction.shape[0]
        if error_est is not None:
            norm = self.normalizer(error_est) + self.beta
        else:
            norm = np.ones(n_test)
        if prediction.ndim > 1:
            ret_val = self.error_func.apply(prediction, y)
        else:
            ret_val = self.error_func.apply(prediction, y) / norm
        return ret_val

    def predict(self,
                prediction,
                nc,
                significance,
                error_est=None):
        n_test = prediction.shape[0]
        if error_est is not None:
            norm = self.normalizer(error_est) + self.beta
        else:
            norm = np.ones(n_test)

        intervals = np.zeros((n_test, 2))
        err_dist = self.error_func.apply_inverse(nc, significance)
        err_dist = np.hstack([err_dist] * n_test)
        if prediction.ndim > 1:
            intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
            intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
        else:
            err_dist *= norm
            intervals[:, 0] = prediction - err_dist[0, :]
            intervals[:, 1] = prediction + err_dist[1, :]
        return intervals

    def eval(self,
             interval,
             y):
        reliability = ((interval[:, 0] < y) * (y < interval[:, 1])).mean()
        efficiency = (np.abs(interval[:, 0]-interval[:, 1])).mean()
        return reliability, efficiency


def conformal_pipeline(significance,
                       calibrate_y,
                       calibrate_prediction,
                       test_prediction,
                       calibrate_error_est=None,
                       test_error_est=None,
                       err_func=AbsErrorErrFunc(),
                       beta=1e-6,
                       normalizer=lambda x: np.exp(x)):
    regressor = RegressorNC(err_func=err_func,
                            normalizer=normalizer,
                            beta=beta)
    nc = regressor.score(calibrate_prediction, calibrate_y, calibrate_error_est)
    interval = regressor.predict(test_prediction, nc, significance, test_error_est)
    return interval





