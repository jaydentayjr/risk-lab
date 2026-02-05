Stress-Aware Portfolio Risk Engine



VaR, Expected Shortfall, Stress Testing \& Monte Carlo Attribution



Overview



This project implements a stress-aware portfolio risk engine designed to measure, attribute, and visualize tail risk under both historical and simulated stress regimes.



Rather than focusing on a single risk metric, the engine follows a full risk workflow:



detect regime changes



measure tail risk severity



attribute losses to assets



replay historical stress



project forward risk using Monte Carlo



communicate results visually



The emphasis is on explainability, robustness, and consistency, mirroring how risk is handled in professional settings.



Key Concepts



Value at Risk (VaR): Loss threshold at a given confidence level



Expected Shortfall (ES): Average loss conditional on exceeding VaR



Rolling risk: Time-varying detection of stress regimes



Risk attribution: Decomposing ES into asset-level contributions



Stress replay: Measuring risk using a known historical stress period



Monte Carlo stress: Forward-looking simulation calibrated to stress



Project Structure

risk\_engine/

├── data/

│   ├── fetch\_stooq.py

│   ├── returns.csv

│

├── models/

│   ├── var\_es.py

│

├── validation/

│   ├── backtesting.py

│

├── attribution/

│   ├── es\_attribution.py

│

├── sim/

│   ├── monte\_carlo.py

│

├── run\_var\_report.py

├── run\_backtest.py

├── run\_rolling\_es.py

├── run\_es\_attribution.py

├── run\_stress\_replay.py

├── run\_mc\_stress.py

├── run\_mc\_es\_attribution.py

│

├── visualize\_rolling\_es.py

├── visualize\_rolling\_component\_es.py

├── visualize\_mc\_es\_share.py

├── visualize\_stress\_equity\_vs\_es.py

│

└── README.md



Data



Daily returns for a multi-asset portfolio:



Equities (SPY, EFA)



Bonds (AGG, TLH)



Commodities (USO, GLD)



Crypto (BTC)



Data sourced via Stooq and aligned into a single return matrix



Portfolio is initially equal-weighted for clarity



Risk Models Implemented

1\. Parametric Models



Gaussian VaR / ES



Student-t VaR / ES (fat tails, configurable degrees of freedom)



2\. Non-Parametric Model



Historical VaR / ES



Validation \& Backtesting



Kupiec unconditional coverage test



Christoffersen independence test



Demonstrates:



Gaussian models pass frequency tests



Gaussian models underestimate tail severity



ES provides a more stable risk signal than VaR



Rolling Risk Analysis



Rolling VaR and ES (60-day window)



Rolling component ES attribution



Shows:



Time-varying tail risk



Diversification weakening during stress



Risk concentration in high-volatility assets



Historical Stress Replay



A known stress window (April–July 2025) is isolated and treated as the risk environment.



Metrics compared:



Full-sample vs stress-window VaR and ES



Worst-day loss



Maximum drawdown



Stress-specific ES attribution



This step grounds the model in realized market stress.



Monte Carlo Stress Simulation



Monte Carlo simulations are calibrated only to the stress window, not the full sample.



Three engines:



Gaussian Monte Carlo



Student-t Monte Carlo



Historical bootstrap Monte Carlo



Simulations produce:



Multi-day (10-day) VaR and ES



Loss distributions



Forward-looking tail severity estimates



Monte Carlo ES Attribution



For simulated worst-case scenarios:



Portfolio losses are decomposed into asset-level contributions



Attribution is additive and exact



Results are compared across models to test robustness



Key insight:



Risk driver ordering is stable across assumptions; fat tails amplify severity rather than reshuffle contributors.



Visualizations



The project emphasizes interpretability:



Rolling ES and component ES



Stacked risk attribution plots



Monte Carlo ES share comparison (Gaussian vs Student-t vs Bootstrap)



Stress equity curve overlaid with rolling ES



These visuals connect risk metrics to realized P\&L intuition.



Core Findings



Tail risk is regime-dependent



Diversification weakens materially during stress



Expected Shortfall captures severity and persistence better than VaR



Macro-sensitive assets dominate stress losses



Fat-tailed models materially increase ES without changing risk ordering



Risk normalizes before P\&L fully recovers



Why This Is a Risk Engine (Not a Toy Model)



This system:



Detects stress regimes



Attributes tail risk dynamically



Grounds assumptions in historical stress



Projects forward losses under uncertainty



Produces explainable, decision-ready outputs



It mirrors how real-world risk teams think about portfolio risk.



Possible Extensions



Portfolio reweighting and capital allocation



Scenario-specific stress calibration



Correlation stress testing



Liquidity-adjusted risk



Reporting dashboards

