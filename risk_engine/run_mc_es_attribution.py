import numpy as np
import pandas as pd

from risk_engine.sim.monte_carlo import (
    simulate_gaussian_mc,
    simulate_student_t_mc,
    simulate_bootstrap_mc,
    var_es_from_losses,
)


def mc_es_attribution(asset_paths: np.ndarray, weights: np.ndarray, alpha: float, asset_names: list[str]):
    """
    asset_paths: (n_sims, horizon, n_assets) simulated daily asset returns
    weights: (n_assets,)
    Returns:
      VaR, ES, component_ES (Series), share (Series)
    """
    # Linear horizon P&L approximation:
    # asset_contrib_return_s = sum_t w_i * r_{s,t,i}
    asset_contrib = (asset_paths * weights.reshape(1, 1, -1)).sum(axis=1)  # (n_sims, n_assets)

    # Portfolio horizon return = sum_i asset_contrib_i
    port_h = asset_contrib.sum(axis=1)  # (n_sims,)
    losses = -port_h  # positive = loss

    # VaR/ES on portfolio losses
    var, es = var_es_from_losses(losses, alpha)

    # Tail scenarios (losses beyond VaR threshold)
    tail = losses >= var
    if tail.sum() < 5:
        raise ValueError("Too few tail scenarios. Increase n_sims or lower alpha.")

    # Component ES per asset: mean(loss contribution | tail)
    # loss contribution per asset = -asset_contrib_return
    comp_es = (-asset_contrib[tail, :]).mean(axis=0)  # (n_assets,)

    comp_es = pd.Series(comp_es, index=asset_names).sort_values(ascending=False)
    share = (comp_es / es).sort_values(ascending=False)

    return var, es, comp_es, share


def main():
    # ---- Settings (match your MC stress run) ----
    alpha = 0.95
    stress_start = "2025-04-01"
    stress_end = "2025-07-01"

    horizon = 10
    n_sims = 50_000
    seed = 42

    # Student-t df (same as before)
    df_t = 6.0

    # ---- Load stress window returns to calibrate ----
    asset_rets = pd.read_csv("returns.csv", parse_dates=[0], index_col=0)
    asset_rets.index.name = "Date"

    stress_assets = asset_rets.loc[stress_start:stress_end].dropna()
    if stress_assets.empty:
        raise ValueError("No data in stress window. Check dates or returns.csv range.")

    asset_names = list(stress_assets.columns)

    # Equal weights (same as your project so far)
    n_assets = stress_assets.shape[1]
    w = np.ones(n_assets) / n_assets

    # Stress-calibrated mean/cov
    mu = stress_assets.mean(axis=0).values
    cov = stress_assets.cov().values

    rng = np.random.default_rng(seed)

    # ---- Simulate paths ----
    paths_g = simulate_gaussian_mc(mu, cov, n_sims=n_sims, horizon=horizon, rng=rng)
    paths_t = simulate_student_t_mc(mu, cov, df=df_t, n_sims=n_sims, horizon=horizon, rng=rng)
    paths_b = simulate_bootstrap_mc(stress_assets, n_sims=n_sims, horizon=horizon, rng=rng)

    results = []

    for model_name, paths in [
        ("Gaussian_MC", paths_g),
        (f"StudentT_MC_df{df_t:g}", paths_t),
        ("Bootstrap_MC", paths_b),
    ]:
        var, es, comp_es, share = mc_es_attribution(paths, w, alpha, asset_names)

        print(f"\n=== MC ES Attribution: {model_name} (alpha={alpha}, horizon={horizon}d) ===")
        print(f"VaR: {var:.6f} | ES: {es:.6f}")
        print("\nTop contributors (component ES):")
        print(comp_es.head(10).round(6))

        # Save tidy rows
        for a in asset_names:
            results.append({
                "model": model_name,
                "alpha": alpha,
                "horizon_days": horizon,
                "asset": a,
                "weight": float(w[asset_names.index(a)]),
                "component_ES": float(comp_es.get(a, 0.0)),
                "share_of_ES": float(share.get(a, 0.0)),
            })

    out = pd.DataFrame(results)
    out.to_csv("mc_stress_es_attribution.csv", index=False)

    print("\nSaved: mc_stress_es_attribution.csv")
    print("Tip: sort by share_of_ES within each model to see concentration.")


if __name__ == "__main__":
    main()
