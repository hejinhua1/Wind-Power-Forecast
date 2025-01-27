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


def CRPS(observations, forecasts, m=10000):
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    n_samples = forecasts.shape[0]

    if observations.ndim == forecasts.ndim - 1:
        assert observations.shape == forecasts.shape[1:]
        observations = observations[np.newaxis, ...]

        # Calculate MAE between forecasts and observations
        mae_score = np.mean(np.abs(forecasts - observations))

        # Estimate E|X - X'| using random sampling
        i_idx = np.random.randint(n_samples, size=m)
        j_idx = np.random.randint(n_samples, size=m)
        diff_samples = np.abs(forecasts[i_idx] - forecasts[j_idx])
        diff_score = np.mean(diff_samples)

        crps_value = mae_score - 0.5 * diff_score
        return crps_value
    elif observations.ndim == forecasts.ndim:
        return np.mean(np.abs(observations - forecasts))


def ES(observations, forecasts, m=10000):
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    n_samples = forecasts.shape[0]

    if observations.ndim == forecasts.ndim - 1:
        assert observations.shape == forecasts.shape[1:]
        observations = observations[np.newaxis, ...]

        # Calculate score term
        X_Y = forecasts - observations
        score = np.mean(np.linalg.norm(X_Y, axis=(1, 2)))

        # Estimate E||X - X'|| using random sampling
        i_idx = np.random.randint(n_samples, size=m)
        j_idx = np.random.randint(n_samples, size=m)
        forecasts_diff = forecasts[i_idx] - forecasts[j_idx]
        norm_diff = np.linalg.norm(forecasts_diff, axis=(1, 2))
        diff_score = np.mean(norm_diff)

        es_value = score - 0.5 * diff_score
        return es_value
    elif observations.ndim == forecasts.ndim:
        return np.linalg.norm(observations - forecasts)


def VS(observations, forecasts, m=10000):
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    n_samples, n_obs, n_timesteps = forecasts.shape

    # Generate random indices for (i, t, s) triplets
    i_indices = np.random.randint(n_obs, size=m)
    t_indices = np.random.randint(n_timesteps, size=m)
    s_indices = np.random.randint(n_timesteps, size=m)

    # Compute observation differences
    obs_diff = np.abs(observations[i_indices, t_indices] - observations[i_indices, s_indices]) ** 0.5

    # Compute forecast differences and their mean
    pred_diff = np.abs(
        forecasts[:, i_indices, t_indices] - forecasts[:, i_indices, s_indices]
    ) ** 0.5
    pred_diff_mean = np.mean(pred_diff, axis=0)

    # Calculate VS value
    vs_value = np.mean((obs_diff - pred_diff_mean) ** 2)
    return vs_value


if __name__ == '__main__':
    np.random.seed(2024)

    # Test Case 1
    print("Test Case 1: Predictions from N(0,1), Observations are zeros")
    n_samples = 10000  # Increased sample size for demonstration
    n_obs, n_timesteps = 2, 3
    observations = np.zeros((n_obs, n_timesteps))
    forecasts = np.random.normal(0, 1, (n_samples, n_obs, n_timesteps))

    crps = CRPS(observations, forecasts, m=100000)
    es = ES(observations, forecasts, m=100000)
    vs = VS(observations, forecasts, m=100000)
    print(f"CRPS: {crps:.4f}")  # Theoretical ~0.234
    print(f"ES: {es:.4f}")  # Theoretical ~0.717
    print(f"VS: {vs:.4f}")  # Depends on implementation

    # Test Case 2
    print("\nTest Case 2: Perfect forecasts")
    forecasts_perfect = np.tile(observations, (n_samples, 1, 1))
    print(f"CRPS (Perfect): {CRPS(observations, forecasts_perfect):.4f}")
    print(f"ES (Perfect): {ES(observations, forecasts_perfect):.4f}")
    print(f"VS (Perfect): {VS(observations, forecasts_perfect):.4f}")