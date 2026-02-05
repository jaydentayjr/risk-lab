import pandas as pd

from risk_engine.attribution.es_attribution import (
    es_attribution_historical,
    rolling_es_attribution_historical,
)


def main():
    # Use your existing returns matrix (asset returns, not portfolio returns)
    # If your file is returns.csv with Date column, we read it like this:
    asset_rets = pd.read_csv("returns.csv", parse_dates=[0], index_col=0)
    asset_rets.index.name = "Date"

    # Equal weights for now (same as before)
    n = asset_rets.shape[1]
    weights = pd.Series(1.0 / n, index=asset_rets.columns)

    alpha = 0.95
    window = 60

    # --- Static attribution (whole sample) ---
    res = es_attribution_historical(asset_rets, weights, alpha=alpha)

    print(f"\n=== Historical ES Attribution (alpha={alpha}) ===")
    print("Tail obs:", res["tail_count"])
    print("Portfolio ES:", round(res["ES"], 6))
    print("\nComponent ES (should sum ~ ES):")
    comp = res["component_ES"].sort_values(ascending=False)
    print(comp.round(6))
    print("\nSum component ES:", round(float(comp.sum()), 6))

    # Save static table
    static = pd.DataFrame({
        "weight": weights,
        "marginal_ES": res["marginal_ES"],
        "component_ES": res["component_ES"],
    })
    static.to_csv("es_attribution_static.csv")

    # --- Rolling attribution ---
    roll = rolling_es_attribution_historical(asset_rets, weights, alpha=alpha, window=window)
    roll.to_csv("es_attribution_rolling.csv")

    print("\nSaved: es_attribution_static.csv")
    print("Saved: es_attribution_rolling.csv")
    print("\nLast rows of rolling attribution:")
    print(roll.tail())


if __name__ == "__main__":
    main()
