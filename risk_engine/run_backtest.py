import numpy as np
import pandas as pd

from risk_engine.models.var_es import var_gaussian
from risk_engine.validation.backtesting import kupiec_test, christoffersen_test
from scipy.stats import t

def rolling_var(returns: pd.Series, alpha: float, window: int=250):
    var_vals = []
    dates = []

    for t in range(window, len(returns)):
        window_returns = returns.iloc[t - window:t]
        var_t = var_gaussian(window_returns, alpha)
        var_vals.append(var_t)
        dates.append(returns.index[t])

    return pd.Series(var_vals, index=dates)

def rolling_var_historical(returns: pd.Series, alpha: float, window: int = 250):
    var_vals = []
    dates = []

    for t in range(window, len(returns)):
        window_returns = returns.iloc[t - window:t]
        q = np.quantile(window_returns.dropna().values, 1 - alpha)  # left tail
        var_vals.append(float(-q))  # positive loss number
        dates.append(returns.index[t])

    return pd.Series(var_vals, index=dates)

def rolling_var_student_t(returns: pd.Series, alpha: float, window: int = 60):
    """
    Rolling student-t parametric VaR.
    Fits a Student-t distribution via MLE on each rolling window and computes VaR.

    VaR is returned as a positive loss number
    """
    var_vals = []
    dates = []

    x = returns.dropna()

    for i in range(window, len(x)):
        w = x.iloc[i - window:i].values

        # Fit Student-t parameters (df, loc, scale)
        # This can occasionally fail or produce weird params for small windows,
        # so we do basic safeguards
        try:
            df, loc, scale = t.fit(w) #MLE fit

            #Safeguards:
            if df <= 2 or scale <= 0 or not np.isfinite([df, loc, scale]).all():
                raise ValueError("unstable t-fit parameters")
        except Exception:
            #Fallback: if fit fails, behave like Gaussian VaR on that window
            mu = w.mean()
            sigma = w.std(ddof=1)
            # Use a large df approximation -> close to normal
            df, loc, scale = 1000.0, mu, sigma

        q = t.ppf(1 - alpha, df, loc=loc, scale=scale) #left tail quantile
        var_vals.append(float(-q)) # VaR as positive loss
        dates.append(x.index[i])

    return pd.Series(var_vals, index=dates, name=f"VaR_t_{alpha}")



def main():
    returns = pd.read_csv(
        "portfolio_returns.csv",
        parse_dates=["Date"],
        index_col="Date"
    )["portfolio_return"]

    alpha = 0.95
    window = 60  # IMPORTANT: you only have ~104 obs, so 250 won't work well

    var_g = rolling_var(returns, alpha, window=window)
    var_h = rolling_var_historical(returns, alpha, window=window)
    var_t = rolling_var_student_t(returns, alpha, window=window)

    for label, var_series in [("Gaussian", var_g), ("Historical", var_h), ("Student-t", var_t)]:
        aligned_r, aligned_v = returns.align(var_series, join="inner")
        breaches = aligned_r < -aligned_v

        print(f"\n=== VaR Backtesting ({label}, alpha={alpha}, window={window}) ===")
        print("Aligned obs:", len(aligned_r))
        print("Breaches:", int(breaches.sum()), "Expected:", (1-alpha)*len(aligned_r))

        kupiec_lr, kupiec_p = kupiec_test(returns, var_series, alpha)
        christ_lr, christ_p = christoffersen_test(returns, var_series)

        print(f"Kupiec LR: {kupiec_lr}, p-value: {kupiec_p}")
        print(f"Christoffersen LR: {christ_lr}, p-value: {christ_p}")


if __name__ == "__main__":
    main()