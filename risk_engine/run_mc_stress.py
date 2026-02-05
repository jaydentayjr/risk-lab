import numpy as np
import pandas as pd

from risk_engine.sim.monte_carlo import (
    simulate_gaussian_mc,
    simulate_student_t_mc,
    simulate_bootstrap_mc,
    portfolio_returns_from_assets,
    var_es_from_losses,
)


def compound_returns(r: np.ndarray) -> np.ndarray:
    """
    r shape: (n_sims, horizon) portfolio simple returns per day
    returns total horizon return per scenario
    """
    return np.prod(1.0 + r, axis=1) - 1.0


def main():
    # ---- Settings ----
    alpha = 0.95
    stress_start = "2025-04-01"
    stress_end = "2025-07-01"

    horizon = 10       # days ahead
    n_sims = 50_000
    seed = 42

    # For Student-t: choose df (fat-tail strength)
    # Lower df => fatter tails. 5-10 is typical stress calibration.
    df_t = 6.0

    # ---- Load returns ----
    asset_rets = pd.read_csv("returns.csv", parse_dates=[0], index_col=0)
    asset_rets.index.name = "Date"

    stress_assets = asset_rets.loc[stress_start:stress_end].dropna()
    if stress_assets.empty:
        raise ValueError("No data in stress window. Check dates or returns.csv range.")

    # Equal weights (same as your project so far)
    n_assets = stress_assets.shape[1]
    w = np.ones(n_assets) / n_assets

    # Stress-calibrated mean/cov (per day)
    mu = stress_assets.mean(axis=0).values
    cov = stress_assets.cov().values

    rng = np.random.default_rng(seed)

    # ---- 1) Gaussian MC ----
    sim_g = simulate_gaussian_mc(mu, cov, n_sims=n_sims, horizon=horizon, rng=rng)
    port_g_daily = portfolio_returns_from_assets(sim_g, w)           # (n_sims, horizon)
    port_g_h = compound_returns(port_g_daily)                        # (n_sims,)
    losses_g = -port_g_h                                             # positive = loss
    var_g, es_g = var_es_from_losses(losses_g, alpha)

    # ---- 2) Student-t MC ----
    sim_t = simulate_student_t_mc(mu, cov, df=df_t, n_sims=n_sims, horizon=horizon, rng=rng)
    port_t_daily = portfolio_returns_from_assets(sim_t, w)
    port_t_h = compound_returns(port_t_daily)
    losses_t = -port_t_h
    var_t, es_t = var_es_from_losses(losses_t, alpha)

    # ---- 3) Bootstrap MC ----
    sim_b = simulate_bootstrap_mc(stress_assets, n_sims=n_sims, horizon=horizon, rng=rng)
    port_b_daily = portfolio_returns_from_assets(sim_b, w)
    port_b_h = compound_returns(port_b_daily)
    losses_b = -port_b_h
    var_b, es_b = var_es_from_losses(losses_b, alpha)

    # ---- Summary ----
    summary = pd.DataFrame(
        [
            {"model": "Gaussian_MC", "alpha": alpha, "horizon_days": horizon, "VaR": var_g, "ES": es_g},
            {"model": f"StudentT_MC_df{df_t:g}", "alpha": alpha, "horizon_days": horizon, "VaR": var_t, "ES": es_t},
            {"model": "Bootstrap_MC", "alpha": alpha, "horizon_days": horizon, "VaR": var_b, "ES": es_b},
        ]
    )

    summary.to_csv("mc_stress_summary.csv", index=False)

    print("\n=== Monte Carlo Stress Simulation Summary ===")
    print(f"Stress window: {stress_start}..{stress_end}")
    print(f"Scenarios: {n_sims:,} | Horizon: {horizon} days | alpha={alpha}")
    print(summary.round(6))
    print("\nSaved: mc_stress_summary.csv")

    # Optional: save losses for later plotting (can be large)
    out = pd.DataFrame({"loss_gauss": losses_g, "loss_t": losses_t, "loss_boot": losses_b})
    out.sample(5000, random_state=1).to_csv("mc_stress_losses_sample.csv", index=False)
    print("Saved: mc_stress_losses_sample.csv (5,000 sampled rows for plotting)")


if __name__ == "__main__":
    main()
