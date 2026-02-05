import numpy as np
import pandas as pd

from risk_engine.models.var_es import (
    var_gaussian, es_gaussian,
    var_historical, es_historical,
    es_student_t
    )

from scipy.stats import t

def rolling_metrics_gaussian(r: pd.Series, alpha: float, window: int):
    dates, var_list, es_list = [], [], []
    for i in range(window, len(r)):
        w = r.iloc[i-window:i]
        var_list.append(var_gaussian(w, alpha))
        es_list.append(es_gaussian(w, alpha))
        dates.append(r.index[i])
    return pd.DataFrame({"VaR": var_list, "ES": es_list}, index=dates)


def rolling_metrics_historical(r: pd.Series, alpha: float, window: int):
    dates, var_list, es_list = [], [], []
    for i in range(window, len(r)):
        w = r.iloc[i-window:i]
        var_list.append(var_historical(w, alpha))
        es_list.append(es_historical(w, alpha))
        dates.append(r.index[i])
    return pd.DataFrame({"VaR": var_list, "ES": es_list}, index=dates)


def rolling_metrics_student_t(r: pd.Series, alpha: float, window: int):
    dates, var_list, es_list = [], [], []
    for i in range(window, len(r)):
        w = r.iloc[i-window:i]
        # VaR from fitted Student-t
        x = w.dropna().values
        try:
            df, loc, scale = t.fit(x)
            if df <= 2 or scale <= 0 or not np.isfinite([df, loc, scale]).all():
                raise ValueError("unstable t-fit")
            q = t.ppf(1 - alpha, df, loc=loc, scale=scale)
            var_val = float(-q)
        except Exception:
            var_val = var_gaussian(w, alpha)

        es_val = es_student_t(w, alpha)

        var_list.append(var_val)
        es_list.append(es_val)
        dates.append(r.index[i])

    return pd.DataFrame({"VaR": var_list, "ES": es_list}, index=dates)

def main():
    r = pd.read_csv(
        "portfolio_returns.csv",
        parse_dates=["Date"],
        index_col="Date"
    )["portfolio_return"]

    alpha = 0.95
    window = 60

    g = rolling_metrics_gaussian(r, alpha, window).rename(columns={"VaR": "VaR_gauss", "ES": "ES_gauss"})
    h = rolling_metrics_historical(r, alpha, window).rename(columns={"VaR": "VaR_hist", "ES": "ES_hist"})
    tt = rolling_metrics_student_t(r, alpha, window).rename(columns={"VaR": "VaR_t", "ES": "ES_t"})

    out = g.join(h, how="inner").join(tt, how="inner")
    out.to_csv("rolling_var_es.csv")

    print("\nSaved: rolling_var_es.csv")
    print(out.tail())


if __name__ == "__main__":
    main()