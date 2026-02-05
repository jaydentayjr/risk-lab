import numpy as np
import pandas as pd
from scipy.stats import chi2

def kupiec_test(returns: pd.Series, var_series: pd.Series, alpha: float):
    """
    Kupiec unconditional coverage test.
    H0: breach frequency == (1 - alpha)
    """
    returns = returns.align(var_series, join="inner")[0]
    var_series = var_series.loc[returns.index]

    aligned_r, aligned_v = returns.align(var_series, join="inner")
    breaches = aligned_r < -aligned_v
    
    breaches = returns < -var_series
    x = breaches.sum()
    n = len(breaches)

    p = 1 - alpha
    phat = x/n if n > 0 else 0

    if x == 0 or x == n:
        return np.nan, np.nan
    
    lr = -2 * (
        np.log(((1 - p) ** (n - x)) * (p ** x))
        - np.log(((1 - phat) ** (n - x)) * (phat ** x))
    )

    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value

def christoffersen_test(returns: pd.Series, var_series: pd.Series):
    """
    Christoffersen independence test.
    Tests whether breaches are independent over time.
    """
    returns = returns.align(var_series, join="inner")[0]
    var_series = var_series.loc[returns.index]

    aligned_r, aligned_v = returns.align(var_series, join="inner")
    breaches = aligned_r < -aligned_v

    breaches = (returns < -var_series).astype(int)

    n00 = n01 = n10 = n11 = 0
    for t in range (1, len(breaches)):
        prev, curr = breaches.iloc[t - 1], breaches.iloc[t]
        if prev == 0 and curr == 0: n00 += 1
        if prev == 0 and curr == 1: n01 += 1
        if prev == 1 and curr == 0: n10 += 1
        if prev == 1 and curr == 1: n11 += 1
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi in [0,1] or pi0 in [0,1] or pi1 in [0,1]:
        return np.nan, np.nan
    
    ll_ind = (
        (n00 + n01) * np.log(1 - pi)
        + (n10 + n11) * np.log(pi)
    )
    ll_dep = (
        n00 * np.log(1 - pi0) +
        n01 * np.log(pi0) +
        n10 * np.log(1 - pi1) +
        n11 * np.log(pi1)
    )

    lr = -2 * (ll_ind - ll_dep)
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value