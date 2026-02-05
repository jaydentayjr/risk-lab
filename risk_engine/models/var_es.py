import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute portfolio returns r_p(t) = sum_i w_i * r_i(t)
    returns: DataFrame (T x N)
    weights: ndarray (N,)
    """
    if returns.shape[1] != len(weights):
        raise ValueError("weights length must match number of columns in returns")
    
    w = np.array(weights, dtype=float)
    w = w / w.sum() #normalise
    rp = returns.values @ w
    return pd.Series(rp, index=returns.index, name="portfolio_return")


# -----------------------
# Parametric (Gaussian)
# -----------------------
def var_gaussian(rp: pd.Series, alpha: float) -> float:
    """
    VaR at level alpha (e.g. 0.99) as a positive number (loss).
    Uses Normal(mu, sigma)
    """
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    q = norm.ppf(1 - alpha, loc=mu, scale=sigma) #left tail quantile
    return(float(-q))

def es_gaussian(rp: pd.Series, alpha:float) -> float:
    """
    ES at level alpha as a postivie number (loss)
    For Normal, ES has closed form
    """
    mu = rp.mean()
    sigma = rp.std(ddof=1)
    z = norm.ppf(1 - alpha)
    es = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    return float(es)

# Historical
def var_historical(rp: pd.Series, alpha: float) -> float:
    q = np.quantile(rp.dropna().values, 1 - alpha)
    return float(-q)

def es_historical(rp: pd.Series, alpha: float) -> float:
    """
    ES = average of losses beyond VaR threshold (left tail)
    """
    x = rp.dropna().values
    q = np.quantile(x, 1-alpha)
    tail = x[x <= q]
    return float(-tail.mean())

def es_student_t(rp: pd.Series, alpha: float) -> float:
    """
    Parametric ES under Student-t fitted by MLE.
    Returns ES as a positive loss number.
    """
    x = rp.dropna().values
    df, loc, scale = student_t.fit(x)

    # Guardrails
    if df <= 2 or scale <= 0 or not np.isfinite([df, loc, scale]).all():
        # fallback to Gaussian ES using sample mean/std
        return es_gaussian(rp, alpha)

    # Left-tail quantile in return space (negative for losses)
    q = student_t.ppf(1 - alpha, df, loc=loc, scale=scale)

    # Standardized quantile
    z = (q - loc) / scale
    fz = student_t.pdf(z, df)

    # ES of RETURNS in the left tail (this will be negative)
    es_return = loc - scale * ((df + z**2) / (df - 1)) * (fz / (1 - alpha))

    # Convert to positive loss
    return float(-es_return)