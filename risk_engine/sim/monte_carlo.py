import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def portfolio_returns_from_assets(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # asset_returns shape: (n_sims, n_assets) or (n_sims, horizon, n_assets)
    return np.tensordot(asset_returns, weights, axes=([-1], [0]))


def var_es_from_losses(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    losses: positive = loss, negative = gain
    Returns (VaR, ES) as positive numbers.
    """
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    es = float(tail.mean()) if len(tail) > 0 else float("nan")
    return float(q), es


def simulate_gaussian_mc(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns simulated asset returns of shape (n_sims, horizon, n_assets).
    Assumes i.i.d. Gaussian increments with mean mu and covariance cov per day.
    """
    n_assets = len(mu)
    # Draw all days in one go: (n_sims*horizon, n_assets)
    x = rng.multivariate_normal(mean=mu, cov=cov, size=n_sims * horizon)
    return x.reshape(n_sims, horizon, n_assets)


def simulate_student_t_mc(
    mu: np.ndarray,
    cov: np.ndarray,
    df: float,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Multivariate Student-t via Gaussian scale mixture:
      X = mu + sqrt(df / U) * Z
      where Z ~ N(0, cov), U ~ Chi2(df)

    Returns shape (n_sims, horizon, n_assets).
    """
    n_assets = len(mu)
    z = rng.multivariate_normal(mean=np.zeros(n_assets), cov=cov, size=n_sims * horizon)
    u = rng.chisquare(df=df, size=n_sims * horizon)
    scales = np.sqrt(df / u)  # (n_sims*horizon,)
    x = mu + (z.T * scales).T
    return x.reshape(n_sims, horizon, n_assets)


def simulate_bootstrap_mc(
    stress_returns: pd.DataFrame,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blockless bootstrap: sample days with replacement from stress window returns.
    Returns shape (n_sims, horizon, n_assets).
    """
    arr = stress_returns.values
    n_days, n_assets = arr.shape
    idx = rng.integers(low=0, high=n_days, size=(n_sims, horizon))
    return arr[idx, :]
