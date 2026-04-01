# Discussion Section Draft
> For paper Section 5.4 / revised Discussion
> Generated: 2026-04-01

---

## 5.4.1 Why RLMC Outperforms Static Ensemble

The RLMC ensemble achieves statistically significant improvements over the best single-graph model at h=1 (p=0.001) and h=24 (p=0.002), but not at h=6 or h=12 (see Table statistical significance). This non-monotonic pattern is explained by the weight dynamics:

**Table: Mean RLMC ensemble weights (Beijing, best run)**
| Horizon | D (distance) | N (neighbor) | S (similarity) | F (functional) |
|---------|-------------|-------------|----------------|----------------|
| 1h  | 0.001 | 0.373 | 0.304 | 0.323 |
| 6h  | 0.005 | 0.588 | 0.128 | 0.280 |
| 12h | 0.002 | 0.437 | 0.002 | 0.560 |
| 24h | 0.091 | 0.396 | 0.280 | 0.235 |

At 1h and 24h, the RLMC agent learns to strongly differentiate among the four graph types: at 1h, three graphs (N, S, F) each contribute meaningfully, creating a diversified ensemble. At 24h, N remains dominant but D weight rises to 9.1%, suggesting that geographic distance begins mattering at diurnal timescales when synoptic-scale transport patterns emerge. This diversity of weighting yields measurable error reduction.

At 6h and 12h, N dominates (58.8% and 43.7%), leaving less opportunity for ensemble diversity gains — the single best model (N) is already capturing most of the signal. The agent's difficulty in achieving uniform differentiation limits RLMC's advantage at these horizons.

**Design implication**: Dynamic ensemble weighting adds most value when no single graph structure is clearly dominant — either because multiple graph types carry complementary information (short horizon) or because the predictive value of each graph shifts with temporal scale (long horizon).

---

## 5.4.2 Graph Type Importance Analysis

The near-zero contribution of the Distance graph (D) across all horizons is counterintuitive: geographic proximity is typically assumed to be the primary driver of air quality correlation. We propose two explanations:

1. **Wind channel effects**: Beijing's monitoring stations are positioned along prevailing SW-NE wind corridors. Station connectivity through the Neighbor graph (radius=45 km) captures wind-corridor adjacency better than the isotropic distance kernel, explaining N's dominance.

2. **Gaussian kernel saturation**: The distance threshold (σ=16 km, ε=0.4) creates a very sparse connectivity for Beijing stations (inter-station distances range from 8–95 km). Most pairs below 40 km have near-zero weights after thresholding, making D structurally similar to N with less coverage.

The Functional graph (F/POI) spikes at h=12 (56.0%): stations sharing similar POI emission environments (industrial zones, residential areas, traffic corridors) exhibit correlated 12-hour pollution build-up patterns that pure spatial models miss. This intermediate timescale corresponds to the duration of a typical synoptic event (frontal passage) — a physically interpretable signal.

---

## 5.4.3 Cross-Region Generalization (Delhi NCT)

DMGENet generalizes to Delhi NCT despite three key differences from Beijing: (1) 3× higher average PM2.5 (168 vs 82 μg/m³), (2) different meteorological regime (monsoon-dominated vs continental), (3) different dataset era (2020–2023 vs 2013–2017).

**Key finding**: At h=12 and h=24, Delhi achieves *lower error* than Beijing despite higher absolute concentrations. We attribute this to Delhi's stronger diurnal and seasonal pollution cycles — driven by consistent agricultural burning patterns and temperature-driven boundary layer variations — which GNN spatial structures can capture more reliably.

Strikingly, the RLMC agent learns **completely different** weight profiles for Delhi vs. Beijing:

**Table: Delhi NCT mean RLMC ensemble weights**
| Horizon | D | N | S | F |
|---------|---|---|---|---|
| 1h  | 0.002 | 0.453 | 0.043 | 0.504 |
| 6h  | 0.013 | 0.520 | 0.169 | 0.299 |
| 12h | **0.996** | 0.000 | 0.004 | 0.001 |
| 24h | 0.001 | 0.006 | **0.883** | 0.111 |

In Beijing, the Neighbor graph (N) dominates consistently (37–59%). In Delhi, the agent discovers entirely different optimal strategies: at h=12, geographic distance dominates (D=0.996); at h=24, distributional similarity dominates (S=0.883). These extreme but stable weights (confirmed across all 10 runs, MAE std=0.015 for h=12) reflect genuine differences in Delhi's pollution dynamics: Delhi's larger station footprint (spanning ~800 km²) creates stronger distance-based concentration gradients, and Delhi's bimodal seasonal pattern (pre-monsoon/post-monsoon) creates strong distributional similarity across the annual cycle.

This dramatic difference between Beijing and Delhi weight patterns provides strong evidence for the value of adaptive ensemble weighting: a fixed global weighting strategy would fail to capture city-specific pollution dynamics.

---

## 5.4.4 Station-Level Prediction Difficulty

Station-level analysis (1h horizon) reveals systematic difficulty patterns:
- **Hardest stations**: Station 11 (RMSE=19.12 μg/m³), Station 5 (RMSE=17.86), Station 3 (RMSE=17.62)
- **Easiest stations**: Station 6 (RMSE=15.46), Station 0 (RMSE=15.01), Station 9 (RMSE=15.18)

This heterogeneity suggests station-specific local emission sources (traffic, industrial) that create rapid fluctuations unpredictable from network-level spatial patterns. The ASTAM mechanism partially addresses this through node-adaptive attention weighting, but station-specific model tuning remains an open challenge.

---

## 5.4.5 Comparison with Static Ensemble Baselines

From Table 7 (existing paper results), DMGENet consistently outperforms static ensemble methods:
- Average ensemble: treats all four graphs equally — worst because N dominates at most horizons
- GWO/PSO/GA-weighted: static weights optimized globally — cannot adapt to changing conditions
- RLMC: error-conditioned dynamic weights — captures temporal variation in graph relevance

The RL-based approach's advantage is most apparent at longer horizons (24h) where the meteorological context changes significantly over the prediction window, requiring the ensemble to shift from N-dominant (short-range transport) to a more balanced weighting (synoptic mixing).

---

## 5.5 Limitations

1. **Temporal coverage**: The Beijing dataset spans 2013–2017. While we add Delhi 2020–2023 for contemporary validation, the Beijing results may not capture the effects of China's post-2018 air quality improvement policies (Blue Sky Action Plan).

2. **Pollutant specificity**: We validate exclusively on PM2.5. Other pollutants (O₃, NO₂, SO₂) have different reaction kinetics and spatial propagation patterns that may require different graph topologies.

3. **Graph construction heuristics**: Distance thresholds (σ=16 km), neighbor radii (45 km), and similarity thresholds (0.6) are fixed hyperparameters. Learned adaptive graphs (e.g., adaptive graph convolution) would be more principled but computationally heavier.

4. **MDP approximation**: The DDPG ensemble formulation approximates the problem as a Markov Decision Process with a 72-hour state window. Long-range dependencies beyond 72 hours (e.g., multi-day pollution events) are not captured by the state representation.

5. **Computational overhead**: Running four separate GNN models in parallel requires 4× the memory of a single-graph approach. On consumer hardware with limited GPU VRAM, this may require sequential inference.
