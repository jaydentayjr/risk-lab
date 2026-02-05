import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pd.read_csv("mc_stress_es_attribution.csv")

    # Pivot: rows = asset, columns = model, values = share_of_ES
    pivot = (
        df.pivot(index="asset", columns="model", values="share_of_ES")
          .fillna(0.0)
    )

    # Sort assets by Student-t contribution (most conservative model)
    if "StudentT_MC_df6" in pivot.columns:
        pivot = pivot.sort_values(by="StudentT_MC_df6", ascending=False)
    else:
        pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

    assets = pivot.index.tolist()
    models = pivot.columns.tolist()

    x = np.arange(len(assets))
    width = 0.25

    plt.figure(figsize=(14, 6))

    for i, model in enumerate(models):
        plt.bar(
            x + i * width,
            pivot[model],
            width=width,
            label=model
        )

    plt.xticks(x + width, assets)
    plt.ylabel("Share of Expected Shortfall")
    plt.title("Monte Carlo Stress ES Attribution (10-day horizon, α = 0.95)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    plt.savefig("docs/mc_es_share.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
