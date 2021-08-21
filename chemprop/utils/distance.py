from rdkit import DataStructs
from scipy.spatial.distance import *
import rdkit.Chem as Chem
import numpy as np
from .normalize import *
from tqdm import tqdm


def prior_unc(fps_train,
              fps_test,
              fp_fun=None,
              dis_fun=lambda x, y: euclidean(x, y),
              sc_fun=lambda distances: np.min(distances)):
    D = []
    if fp_fun is not None:
        fps_train = [fp_fun(x) for x in fps_train]
        fps_test  = [fp_fun(x) for x in fps_test]
    for i, test_fp in tqdm(enumerate(fps_test)):
        distances = np.array([dis_fun(test_fp, train_fp) for train_fp in fps_train])
        D.append(sc_fun(distances))
    return np.array(D)