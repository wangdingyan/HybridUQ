# https://github.com/yromano/cqr/blob/master/nonconformist/nc.py
import numpy as np


class AbsErrorErrFunc():
    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self,
              prediction,
              y):
        return np.abs(prediction-y)

    def apply_inverse(self,
                      nc,
                      significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size+1))) - 1
        border = min(max(border, 0), nc.size-1)
        return np.vstack([nc[border], nc[border]])


class QuantileRegErrFunc():
    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        error = np.maximum(error_high, error_low)
        return error

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1-significance) * (nc.size+1))) - 1
        index = min(max(index, 0), nc.shape[0]-1)
        return np.vstack([nc[index], nc[index]])


class QuantileRegAsymmetricErrorFunc():
    def __init__(self):
        super(QuantileRegAsymmetricErrorFunc, self).__init__()

    def apply(self,
              prediction,
              y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:,-1]
        error_high = y - y_upper
        error_low = y_lower - y
        error_high = np.reshape(error_high, (y_upper.shape[0], 1))
        error_low = np.reshape(error_low, (y_lower.shape[0], 1))
        return np.concatenate((error_low, error_high), 1)

    def apply_inverse(self,
              nc,
              significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1-significance/2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index,0], nc[index,1]])







