# Stress-Aware Portfolio Risk Engine
VaR / Expected Shortfall - Backtesting - Stress Replay - Monte Carlo Stress - ES Attribution

This repo implements a **stress-aware portfolio risk workflow**:
1) estimate VaR/ES under multiple assumptions (Gaussian / Historical / Student-t)  
2) validate with VaR backtests (Kupiec + Christoffersen)  
3) track **rolling ES** and **rolling component ES** (risk concentration over time)  
4) **replay a historical stress window** and attribute ES drivers  
5) run **stress-calibrated Monte Carlo** and perform **MC ES attribution** (forward-looking risk drivers)

**Key result:** risk driver ordering is robust across models; fat tails increase severity more than they reshuffle drivers.

---

## What’s in here
### Models
- **Gaussian VaR/ES** (parametric, thin tails)
- **Historical VaR/ES** (empirical, data-driven tails)
- **Student-t ES** (fat tails via low degrees of freedom)

### Validation
- **Kupiec test** (unconditional coverage)
- **Christoffersen test** (independence / clustering)

### Stress & Attribution
- **Rolling ES + rolling component ES** (regime detection + concentration)
- **Historical stress replay** (stress window vs full sample)
- **Monte Carlo stress simulation** (Gaussian / Student-t / Bootstrap)
- **Monte Carlo ES attribution** (who drives worst-case simulated losses)

---

## Repo layout
```text
risk_engine/
  attribution/            # ES attribution logic
  data/                   # data fetch (Stooq)
  models/                 # VaR/ES implementations
  sim/                    # Monte Carlo engines
  validation/             # backtesting tests
  run_*.py                # runners (generate outputs)
  visualize_*.py          # plots
outputs/                  # curated example outputs for reviewers
docs/                     # saved figures for README
requirements.txt
