import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(
    "rolling_var_es.csv",
    parse_dates=[0],
    index_col=0
    )

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["ES_gauss"], label="Gaussian ES")
    plt.plot(df.index, df["ES_hist"], label="Historical ES")
    plt.plot(df.index, df["ES_t"], label="Student-t ES")

    plt.title("Rolling Expected Shortfall (95%, window=60)")
    plt.ylabel("Loss (positive)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("docs/rolling_es.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
