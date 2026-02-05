import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(
        "es_attribution_rolling.csv",
        parse_dates=[0],
        index_col=0
    )

    # Keep only component ES columns
    comp_cols = [c for c in df.columns if c.startswith("cES_")]
    comp = df[comp_cols]

    # Rename for cleaner legend
    comp.columns = [c.replace("cES_", "") for c in comp.columns]

    # Split positive and negative contributions
    pos = comp.clip(lower=0)
    neg = comp.clip(upper=0)

    plt.figure(figsize=(13, 6))

    # Positive contributions (stacked above zero)
    plt.stackplot(
        pos.index,
        pos.T,
        labels=pos.columns,
        alpha=0.85
    )

    # Negative contributions (stacked below zero)
    plt.stackplot(
        neg.index,
        neg.T,
        alpha=0.85
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Rolling Component ES (Historical, 95%, window=60)")
    plt.ylabel("ES Contribution")
    plt.xlabel("Date")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.savefig("docs/rolling_component_es.png", dpi=200, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    main()
