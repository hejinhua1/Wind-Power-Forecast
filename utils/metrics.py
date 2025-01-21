import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe




def CRPS(observations, forecasts):
    """
    An alternative but simpler implementation of CRPS for testing purposes
    input:
        observations: np.array of shape (n_timesteps,)
        forecasts: np.array of shape (n_timesteps, n_samples)
    output:
        crps: np.array of shape (n_timesteps,)

    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations[..., np.newaxis]

        score = np.nanmean(abs(forecasts - observations), -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (np.expand_dims(forecasts, -1) - np.expand_dims(forecasts, -2))

        diff_score = np.nanmean(abs(forecasts_diff), axis=(-2, -1))
        crps_value = score - 0.5 * diff_score
        return crps_value
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return abs(observations - forecasts)


import numpy as np


def ES(observations, forecasts):
    """
    Energy Score (ES) implementation
    input:
        observations: np.array of shape (n_timesteps, n_features)
        forecasts: np.array of shape (n_timesteps, n_samples, n_features)
    output:
        es: np.array of shape (n_timesteps,)

    This implementation is based on the identity:

    .. math::
        ES(F, x) = E_F[ ||X - x||_2^2 ] - 1/2 * E_F[ ||X - X'||_2^2 ]

    where X and X' are independent random variables drawn from the forecast distribution F.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis if observations is one less dimension than forecasts
        assert observations.shape == forecasts.shape[:-2]
        observations = observations[..., np.newaxis]

        # Calculate the first expectation E_F[ ||X - Y||_2^2 ]
        score = np.nanmean(np.linalg.norm(forecasts - observations, axis=-1) ** 2, -1)

        # Calculate the second expectation E_F[ ||X - X'||_2^2 ]
        forecasts_diff = np.expand_dims(forecasts, -2) - np.expand_dims(forecasts, -3)
        diff_score = np.nanmean(np.linalg.norm(forecasts_diff, axis=-1) ** 2, axis=(-2, -1))

        # Calculate the final Energy Score
        es_value = score - 0.5 * diff_score
        return es_value

    elif observations.ndim == forecasts.ndim:
        # Handle the case where there is no "realization" axis (deterministic forecast)
        return np.linalg.norm(observations - forecasts, axis=-1) ** 2





if __name__ == '__main__':
    # 模拟一些预测数据和观测数据
    forecasts = np.array([[0.5, 0.7, 0.6], [0.8, 0.7, 1]])
    observations = np.array([0.6, 0.8])

    # 计算 CRPS
    crps_value = CRPS(observations, forecasts)
    print(crps_value)