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
        observations: np.array of shape (n_observation, n_timesteps)
        forecasts: np.array of shape (n_samples, n_observation, n_timesteps)
    output:
        crps: float

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
        assert observations.shape == forecasts.shape[1:]
        observations = observations[np.newaxis, ...]

        mae_score = np.nanmean(abs(forecasts - observations))

        forecasts_diff = (np.expand_dims(forecasts, 0) - np.expand_dims(forecasts, 1))

        diff_score = np.nanmean(abs(forecasts_diff))

        crps_value = mae_score - 0.5 * diff_score

        return crps_value
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return abs(observations - forecasts)



def ES(observations, forecasts):
    """
    Energy Score (ES) implementation
    input:
        observations: np.array of shape (n_observation, n_timesteps)
        forecasts: np.array of shape (n_samples, n_observation, n_timesteps)
    output:
        es: np.array of shape (n_timesteps,)

    This implementation is based on the identity:

    .. math::
        ES(F, x) = E_F[ ||X - x||_2 ] - 1/2 * E_F[ ||X - X'||_2 ]

    where X and X' are independent random variables drawn from the forecast distribution F.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis if observations is one less dimension than forecasts
        assert observations.shape == forecasts.shape[1:]
        observations = observations[np.newaxis, ...]

        # Calculate the first expectation E_F[ ||X - Y||_2 ]
        X_Y = forecasts - observations
        score = np.nanmean(np.linalg.norm(X_Y.flatten()))

        # Calculate the second expectation E_F[ ||X - X'||_2 ]
        forecasts_diff = np.expand_dims(forecasts, 0) - np.expand_dims(forecasts, 1)
        diff_score = np.nanmean(np.linalg.norm(forecasts_diff.flatten()))

        # Calculate the final Energy Score
        es_value = score - 0.5 * diff_score
        return es_value

    elif observations.ndim == forecasts.ndim:
        # Handle the case where there is no "realization" axis (deterministic forecast)
        return np.linalg.norm(observations - forecasts)


def VS(observations, forecasts):
    """
    An alternative but simpler implementation of CRPS for testing purposes
    input:
        observations: np.array of shape (n_observation, n_timesteps)
        forecasts: np.array of shape (n_samples, n_observation, n_timesteps)
    output:
        crps: float

    This implementation is based on the identity:

    .. math::
        VS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

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
    samples = forecasts.shape[0]
    n_observations = forecasts.shape[1]
    timesteps = forecasts.shape[2]
    obser_diff_mat = abs(np.expand_dims(observations, 1) - np.expand_dims(observations, 2)) ** 0.5
    forec_diff_mat = abs(np.expand_dims(forecasts, -1) - np.expand_dims(forecasts, -2)) ** 0.5
    forec_diff_mat_mean = np.mean(forec_diff_mat, axis=0)
    temp = (obser_diff_mat - forec_diff_mat_mean) ** 2
    vs_value = np.mean(temp.flatten())
    return vs_value



if __name__ == '__main__':
    # 模拟一些预测数据和观测数据
    fix_seed = 2024
    np.random.seed(fix_seed)
    observations = np.random.randint(0, 101, size=(3, 2))
    forecasts = np.random.randint(0, 101, size=(4, 3, 2))

    # 计算 CRPS
    crps_value = VS(observations, forecasts)
    print(crps_value)