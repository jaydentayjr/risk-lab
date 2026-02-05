import numpy as np
import pandas as pd

from risk_engine.models.var_es import(
    portfolio_returns,
    var_gaussian,
    es_gaussian,
    var_historical,
    es_historical
)

def main():
    returns = pd.read_csv("returns.csv", parse_dates=["Date"], index_col="Date")

    #Equal weights by default
    n = returns.shape[1]
    w = np.ones(n)/n

    rp = portfolio_returns(returns, w)

    alphas = [0.95, 0.99]
    rows = []
    for a in alphas:
        rows.append({
            "alpha": a,
            "VaR_gaussian": var_gaussian(rp, a),
            "ES_gaussian": es_gaussian(rp, a),
            "VaR_hist": var_historical(rp, a),
            "ES_hist": es_historical(rp, a),
        })

    report = pd.DataFrame(rows).set_index("alpha")
    print("\n Portfolio VaR/ES Report (daily, equal weight)")
    print(report)

    #Save for README
    report.to_csv("var_es_report.csv")
    rp.to_csv("portfolio_returns.csv")
    print("\nSaved: var_es_report.csv,, portfolio_returns.csv")

if __name__ == "__main__":
    main()