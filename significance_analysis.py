# Implementation of a test for stat significant difference in AUC ROC between two binary classifiers
# Implementation is based on code provided by Regis Loeb and correction for correlation between the labels
# produced by two classifiers (they are correlated because compued on the same data)

import numpy as np
import scipy
import pandas as pd


def vs_and_auc(s_pos, s_neg):
    S_delta = np.tile(s_pos.reshape((1, -1)), [len(s_neg), 1]) - np.tile(
        s_neg.reshape((-1, 1)), [1, len(s_pos)]
    )
    mask_larger = (S_delta > 0).astype(int)
    mask_equal = (S_delta == 0).astype(int)
    V_10 = np.sum(mask_larger + 0.5 * mask_equal, axis=0) / len(s_neg)
    V_01 = np.sum(mask_larger + 0.5 * mask_equal, axis=1) / len(s_pos)
    auc = (np.sum(mask_larger) + 0.5 * np.sum(mask_equal)) / (len(s_pos) * len(s_neg))

    return auc, V_10, V_01


def auc_corr(V0_10, V0_01, auc0, V1_10, V1_01, auc1):

    assert len(V0_01) == len(V1_01), print("Something wrong with 01 vectors")
    assert len(V1_10) == len(V0_10), print("Something wrong with 10 vectors")
    W_10 = np.zeros((2, 2))
    W_10[0, 0] = np.dot(V0_10 - auc0, V0_10 - auc0) / (len(V0_10) - 1)
    W_10[1, 1] = np.dot(V1_10 - auc1, V1_10 - auc1) / (len(V1_10) - 1)
    W_10[0, 1] = np.dot(V0_10 - auc0, V1_10 - auc1) / (len(V0_10) - 1)
    W_10[1, 0] = np.dot(V1_10 - auc1, V0_10 - auc0) / (len(V0_10) - 1)

    W_01 = np.zeros((2, 2))
    W_01[0, 0] = np.dot(V0_01 - auc0, V0_01 - auc0) / (len(V0_01) - 1)
    W_01[1, 1] = np.dot(V1_01 - auc1, V1_01 - auc1) / (len(V1_01) - 1)
    W_01[0, 1] = np.dot(V0_01 - auc0, V1_01 - auc1) / (len(V0_01) - 1)
    W_01[1, 0] = np.dot(V1_01 - auc1, V0_01 - auc0) / (len(V0_01) - 1)

    W = W_10 / len(V0_10) + W_01 / len(V0_01)

    return np.nan_to_num(W[0, 1] / np.sqrt(W[0, 0] * W[1, 1]))


def auc_se(auc, num_pos, num_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return np.sqrt(
        (
            auc * (1 - auc)
            + (num_pos - 1) * (q1 - auc ** 2)
            + (num_neg - 1) * (q2 - auc ** 2)
        )
        / (num_pos * num_neg)
    )


def pvalue(auc1, num_pos1, num_neg1, auc2, num_pos2, num_neg2, r):
    se1 = auc_se(auc1, num_pos1, num_neg1)
    se2 = auc_se(auc2, num_pos2, num_neg2)
    z = (auc1 - auc2) / np.sqrt(se1 ** 2 + se2 ** 2 - r * se1 * se2)
    return 1.0 - scipy.stats.norm.cdf(z)


def test_significance(y, yhat0, yhat1, level=0.05):

    s_pos0 = yhat0[y == 1]
    s_neg0 = yhat0[y == 0]
    auc0, V0_10, V0_01 = vs_and_auc(s_pos0, s_neg0)

    s_pos1 = yhat1[y == 1]
    s_neg1 = yhat1[y == 0]
    auc1, V1_10, V1_01 = vs_and_auc(s_pos1, s_neg1)

    r = auc_corr(
        V1_10, V1_01, auc1, V0_10, V0_01, auc0)
    # computes correlation coefficient
    p = pvalue(auc1, len(s_pos1), len(s_neg1), auc0, len(s_pos0), len(s_neg0), r)
    return pd.DataFrame({f'significant': pd.Series(p < level, dtype='int32'), \
								'p_value': pd.Series(p, dtype='float64')})
