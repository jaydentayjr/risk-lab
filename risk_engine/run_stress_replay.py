import pandas as pd
import numpy as np

from risk_engine.models.var_es import(
    var_historical, var_gaussian,
    es_gaussian, es_historical,
    es_student_t
)

from risk_engine.attribution.es_attribution import es_attribution_historical


def cumulative_from_returns(r: pd.Series) -> pd.Series:
    """Convert simple returns to cumulative equity curve starting at 1:0"""
    return(1.0 + r.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) #negative


def summarize_period(label: str, port_ret: pd.Series, alpha: float):
    """Compute core risk metrics for a given return series."""
    v_h = var_historical(port_ret, alpha)
    e_h = es_historical(port_ret, alpha)

    v_g = var_gaussian(port_ret, alpha)
    e_g = es_gaussian(port_ret, alpha)

    e_t = es_student_t(port_ret, alpha)

    eq = cumulative_from_returns(port_ret)
    mdd = max_drawdown(eq)
    worst_day = float(port_ret.min())

    return {
        "label": label,
        "n_obs": int(port_ret.dropna().shape[0]),
        "worst_day_return": worst_day,
        "max_drawdown": mdd,
        "VaR_hist": float(v_h),
        "ES_hist": float(e_h),
        "VaR_gauss": float(v_g),
        "ES_gauss": float(e_g),
        "ES_t": float(e_t),
    }


def main():
    # ---- Inputs ----
    alpha = 0.95

    # Choose the stress window (edit these freely)
    stress_start = "2025-04-01"
    stress_end = "2025-07-01"

    # Load asset returns matrix (same file you used for attribution)
    asset_rets = pd.read_csv("returns.csv", parse_dates=[0], index_col=0)
    asset_rets.index.name = "Date"

    # Equal weights (for now). Later we can load weights from config.
    n = asset_rets.shape[1]
    w = pd.Series(1.0 / n, index=asset_rets.columns)

    # Portfolio returns (consistent with earlier work)
    port = asset_rets.mul(w, axis=1).sum(axis=1).dropna()

    # Slice stress period
    stress = port.loc[stress_start:stress_end]

    if stress.empty:
        raise ValueError(f"No portfolio returns found in stress window {stress_start}..{stress_end}. "
                         f"Check dates in your data.")

    # ---- Summaries ----
    full_summary = summarize_period("FULL SAMPLE", port, alpha)
    stress_summary = summarize_period(f"STRESS REPLAY {stress_start}..{stress_end}", stress, alpha)

    summary_df = pd.DataFrame([full_summary, stress_summary]).set_index("label")
    summary_df.to_csv("stress_replay_summary.csv")

    print("\n=== Stress Replay Summary (alpha=0.95) ===")
    print(summary_df.round(6))
    print("\nSaved: stress_replay_summary.csv")

    # ---- Stress Attribution (Historical ES, in stress window only) ----
    # Attribution should be computed on the same stress window for asset returns.
    stress_assets = asset_rets.loc[stress_start:stress_end].dropna()

    attrib = es_attribution_historical(stress_assets, w, alpha=alpha)
    comp = attrib["component_ES"].sort_values(ascending=False)

    out_attrib = pd.DataFrame({
        "weight": w,
        "marginal_ES": attrib["marginal_ES"],
        "component_ES": attrib["component_ES"],
    }).sort_values("component_ES", ascending=False)

    out_attrib.to_csv("stress_replay_attribution.csv")

    print("\n=== Stress ES Attribution (Historical, alpha=0.95) ===")
    print("Tail obs:", attrib["tail_count"])
    print("Portfolio ES (stress window):", round(attrib["ES"], 6))
    print("\nComponent ES (largest first):")
    print(comp.round(6))
    print("\nSum component ES:", round(float(comp.sum()), 6))
    print("\nSaved: stress_replay_attribution.csv")

    # ---- Save stress equity curve (for easy plotting later) ----
    stress_eq = cumulative_from_returns(stress)
    stress_eq.to_csv("stress_replay_equity_curve.csv", header=["equity"])
    print("\nSaved: stress_replay_equity_curve.csv (equity curve for stress window)")

    # Small extra: save the stress portfolio returns too
    stress.to_csv("stress_replay_portfolio_returns.csv", header=["portfolio_return"])
    print("Saved: stress_replay_portfolio_returns.csv")


if __name__ == "__main__":
    main()