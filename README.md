# Stress-Aware Portfolio Risk Engine
VaR • Expected Shortfall • Stress Testing • Monte Carlo • Risk Attribution

## TL;DR
This project implements a **stress-aware portfolio risk engine** that measures, validates, and explains tail risk across market regimes.

The workflow:
1) estimate VaR / Expected Shortfall under multiple distributional assumptions  
2) validate risk estimates via statistical backtests  
3) track **rolling tail risk and concentration** over time  
4) replay a **historical stress window** and attribute losses  
5) project **forward-looking stress losses** using Monte Carlo simulation  

**Key result:**  
Risk driver ordering is robust across models; fat tails materially increase loss severity but do not reshuffle dominant contributors.

---

## What’s implemented

### Risk models
- **Gaussian VaR / ES** — parametric baseline (thin tails)
- **Historical VaR / ES** — empirical, data-driven tails
- **Student-t ES** — fat-tailed risk via low degrees of freedom

### Validation
- **Kupiec test** — unconditional coverage
- **Christoffersen test** — independence / breach clustering

### Stress & attribution
- **Rolling ES + rolling component ES** — regime detection and concentration
- **Historical stress replay** — stress window vs full sample comparison
- **Monte Carlo stress simulation** — Gaussian / Student-t / Bootstrap
- **Monte Carlo ES attribution** — forward-looking tail-risk drivers

---

## Repository structure
```text
risk_engine/
  attribution/            # ES attribution logic
  data/                   # data fetch (Stooq)
  models/                 # VaR / ES implementations
  sim/                    # Monte Carlo engines
  validation/             # backtesting tests
  run_*.py                # runners (generate outputs)
  visualize_*.py          # plots
outputs/                  # curated example outputs
docs/                     # figures used in README
requirements.txt

---

## Quickstart

> Windows / PowerShell

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

---

## Core workflow

Run the full risk pipeline in order:

```powershell
# 1) Static VaR / ES report
python -m risk_engine.run_var_report

# 2) VaR backtesting (Kupiec + Christoffersen)
python -m risk_engine.run_backtest

# 3) Rolling VaR / ES (time-varying risk)
python -m risk_engine.run_rolling_es

# 4) Historical ES attribution
python -m risk_engine.run_es_attribution

# 5) Historical stress replay
python -m risk_engine.run_stress_replay

# 6) Monte Carlo stress simulation
python -m risk_engine.run_mc_stress
python -m risk_engine.run_mc_es_attribution

---

## How to add Visual diagnostics (Markdown example)

Paste **after Core workflow**:

```markdown
---

## Visual diagnostics

Generate key plots used for analysis and reporting:

```powershell
python -m risk_engine.visualize_rolling_es
python -m risk_engine.visualize_rolling_component_es
python -m risk_engine.visualize_mc_es_share
python -m risk_engine.visualize_stress_equity_vs_es


```markdown
### Monte Carlo ES share by asset
![MC ES Share](docs/mc_es_share.png)
