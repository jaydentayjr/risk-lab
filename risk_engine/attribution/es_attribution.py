import numpy as np
import pandas as pd


def _tail_mask(port_ret: pd.Series, alpha: float) -> pd.Series:
    """
    Returns a boolean mask for tail days: port_ret <= empirical (1-alpha) quantile.
    """
    q = port_ret.quantile(1 - alpha)
    return port_ret <= q


def es_attribution_historical(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
) -> dict:
    """
    Historical ES attribution using tail conditioning on portfolio returns.

    Returns dict with:
      - ES (positive)
      - marginal_ES (Series, positive loss per unit weight)
      - component_ES (Series, positive loss contribution)
      - tail_count
      - tail_threshold (quantile return)
    """
    # Align weights to columns
    weights = weights.reindex(asset_returns.columns).astype(float)

    port_ret = asset_returns.mul(weights, axis=1).sum(axis=1)

    q = port_ret.quantile(1 - alpha)
    tail = port_ret <= q
    tail_count = int(tail.sum())

    if tail_count < 2:
        raise ValueError("Too few tail observations to estimate ES attribution.")

    # ES as positive loss
    es = float(-port_ret[tail].mean())

    # Marginal ES_i = -E[r_i | tail]
    mES = -asset_returns.loc[tail].mean(axis=0)

    # Component ES_i = w_i * mES_i
    cES = weights * mES

    return {
        "ES": es,
        "marginal_ES": mES,
        "component_ES": cES,
        "tail_count": tail_count,
        "tail_threshold": float(q),
    }


def rolling_es_attribution_historical(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling historical ES attribution.
    Returns a DataFrame indexed by date with:
      - ES (portfolio)
      - component ES per asset (one column per asset)
    """
    weights = weights.reindex(asset_returns.columns).astype(float)

    rows = []
    idx = []

    for t in range(window, len(asset_returns)):
        wret = asset_returns.iloc[t - window : t]
        port_ret = wret.mul(weights, axis=1).sum(axis=1)

        q = port_ret.quantile(1 - alpha)
        tail = port_ret <= q

        if int(tail.sum()) < 2:
            # Skip windows with insufficient tail points
            continue

        es = float(-port_ret[tail].mean())
        mES = -wret.loc[tail].mean(axis=0)
        cES = weights * mES

        row = {"ES": es}
        row.update({f"cES_{col}": float(cES[col]) for col in asset_returns.columns})

        rows.append(row)
        idx.append(asset_returns.index[t])

    out = pd.DataFrame(rows, index=pd.Index(idx, name="Date"))
    return out
