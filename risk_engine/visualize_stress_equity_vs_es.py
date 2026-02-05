import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ---- Load stress equity curve ----
    equity = pd.read_csv(
        "stress_replay_equity_curve.csv",
        parse_dates=[0],
        index_col=0
    )
    equity.columns = ["equity"]

    # Convert equity to drawdown
    peak = equity["equity"].cummax()
    drawdown = equity["equity"] / peak - 1.0

    # ---- Load rolling ES ----
    es = pd.read_csv(
        "rolling_var_es.csv",
        parse_dates=[0],
        index_col=0
    )

    # Use historical ES (most interpretable)
    es_series = es["ES_hist"]

    # Align dates
    dd, es_series = drawdown.align(es_series, join="inner")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Drawdown (left axis)
    ax1.plot(
        dd.index,
        dd.values,
        color="red",
        linewidth=2,
        label="Stress Drawdown"
    )
    ax1.set_ylabel("Drawdown", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.axhline(0, color="black", linewidth=0.8)

    # ES (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
        es_series.index,
        es_series.values,
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Rolling ES (95%)"
    )
    ax2.set_ylabel("Expected Shortfall", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Titles and layout
    plt.title("Stress Equity Curve vs Rolling Expected Shortfall")
    fig.legend(loc="upper right")
    plt.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()

    plt.savefig("docs/stress_equity_vs_es.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
