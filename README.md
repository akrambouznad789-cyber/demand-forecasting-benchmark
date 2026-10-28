# Probabilistic Demand Forecasting: Time-Series Foundation Models vs. Strong Baselines

**Status: in progress** (started 2026-10-27) · Akram Bouznad, Polytechnique Montréal

## Research question
Do zero-shot time-series foundation models (Chronos, TimesFM, TabPFN-TS) forecast retail demand better than
classical baselines, and are their prediction intervals calibrated enough to set inventory on?

## Hypotheses
- **H1.** On aggregated store × department series (smooth, strongly weekly), foundation models beat the seasonal
  naive and ETS baselines on MASE and on the scaled quantile loss.
- **H2.** On item-level series (intermittent demand, many zero-sale days), their advantage shrinks or reverses.
- **H3.** Their 80% prediction intervals are miscalibrated (coverage far from 0.80) on intermittent series,
  which matters for safety stock.

## Protocol
- **Data:** M5 (Walmart daily unit sales, 2011–2016, 3 states, 10 stores). First level studied: store × department,
  70 daily series. Next: item-level series.
- **Test window:** the last 28 days (2016-04-25 to 2016-05-22), the M5 horizon. Next: several rolling origins.
- **Metrics:** MASE for the point forecast; quantile (pinball) loss at 0.1, 0.5 and 0.9, scaled like MASE
  (twice it approximates the CRPS); coverage of the 80% interval.
- **Models:** seasonal naive with empirical quantiles (done); ETS and LightGBM (next); Chronos, TimesFM and
  TabPFN-TS in zero-shot mode (PyTorch, GPU).

## Results so far
| Model | Series | MASE | Scaled quantile loss | 80% interval coverage |
|---|---|---|---|---|
| Seasonal naive + empirical quantiles | 70 | 1.150 | 0.364 | 0.807 |

The baseline's intervals are well calibrated on average (0.807 for a 0.80 target), but not everywhere:
CA_2_FOODS_2 has the worst MASE (2.401), and WI_2_FOODS_1 (MASE 2.091) has only 0.393 coverage.
These FOODS series are the first cases for the error analysis.

## Roadmap
- [x] Data pipeline, metrics, seasonal naive baseline with empirical quantiles
- [ ] ETS and LightGBM baselines
- [ ] Zero-shot Chronos, TimesFM and TabPFN-TS (PyTorch, Colab GPU)
- [ ] Rolling origins; item-level intermittent series
- [ ] Experiment tracking (TensorBoard); Docker image tested end to end
- [ ] Error analysis and a short technical report

## Run
```
pip install -r requirements.txt
python run_benchmark.py
```
The script downloads the M5 files (about 50 MB) into `data/raw/` and writes `results/scores_*.csv` and
`results/forecasts_*.csv`. A `Dockerfile` is included (not yet tested end to end).

## Data
M5 Forecasting competition data (Walmart, via the University of Nicosia and Kaggle), downloaded from Nixtla's
public mirror (`github.com/Nixtla/m5-forecasts`).
