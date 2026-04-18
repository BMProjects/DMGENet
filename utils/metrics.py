import numpy as np
from sklearn.metrics import r2_score


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def R2(pred, true):
    return r2_score(pred, true)

def R2_new(pred, true):
    """Per-horizon R^2 for a [samples, N, T_out] array; matches sklearn.r2_score."""
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError("Inputs must be 3-D arrays of shape [samples, N, T_out]")

    _, _, T_out = pred.shape
    r2_list = []
    for t in range(T_out):
        y_true_t = true[:, :, t].reshape(-1)
        y_pred_t = pred[:, :, t].reshape(-1)
        r2_list.append(r2_score(y_true_t, y_pred_t))
    return np.array(r2_list)


def index_of_agreement(observed, predicted):
    """Willmott's Index of Agreement (IA). Returns a scalar in (-∞, 1]; 1 is perfect."""
    observed = np.array(observed)
    predicted = np.array(predicted)

    mean_observed = np.mean(observed)

    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((np.abs(observed - mean_observed) + np.abs(predicted - mean_observed)) ** 2)

    ia = 1 - (numerator / denominator)

    return ia

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2(pred, true)

    return mae, rmse, mape, r2

def metric_new(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2_new(pred, true)

    return mae, rmse, mape, r2

def metric_multi_sites(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    ia = index_of_agreement(true, pred)
    r2 = R2(pred.reshape(-1), true.reshape(-1))

    return mae, rmse, ia, r2