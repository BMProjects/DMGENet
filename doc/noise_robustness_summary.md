# Noise Robustness Summary (for Paper Section 5.X)
> Data from: doc/noise_robustness.csv
> Average across 4 graph models, Beijing 12-station, best base model

## Results Table

| Horizon | 0% noise | 5% noise | 10% noise | 15% noise | 20% noise |
|---------|----------|----------|-----------|-----------|-----------|
| 1h  RMSE | 16.7 | 33.2 (+99%) | 54.3 (+226%) | 71.6 (+330%) | 85.1 (+411%) |
| 6h  RMSE | 37.0 | 43.5 (+18%) | 56.2 (+52%)  | 68.8 (+86%)  | 78.4 (+112%) |
| 12h RMSE | 49.5 | 52.4 (+6%)  | 62.9 (+27%)  | 74.5 (+51%)  | 82.5 (+67%)  |
| 24h RMSE | 63.4 | 64.6 (+2%)  | 69.4 (+10%)  | 75.7 (+19%)  | 81.2 (+28%)  |

## Interpretation

The model exhibits a strong **horizon-dependent robustness pattern**:

- **Short-term (h=1)**: Highly sensitive (+99% RMSE at just 5% noise). Short-term prediction relies on precise feature values for immediate-term extrapolation — noise directly degrades precision.

- **Long-term (h=24)**: Remarkably robust (+2% RMSE at 5% noise, only +28% at 20%). Long-horizon prediction relies on smoothed temporal patterns captured across the 72-hour input window, which naturally averages out measurement noise.

This behavior is consistent with the **temporal aggregation** properties of TCN: longer receptive fields smooth out local noise perturbations, providing implicit regularization at longer horizons.

## Practical Implications

For deployment in real monitoring networks where sensor noise and calibration drift are common (typically 5-15% of signal), DMGENet remains reliable for 12h and 24h forecasting (RMSE increase < 51%) but should be used with caution for 1h forecasts unless data quality is verified.

**Note**: The 5% noise level corresponds to typical sensor calibration error in low-cost PM2.5 sensors (e.g., PurpleAir). Official monitoring stations (Beijing CNEMC network used in this paper) typically achieve < 2% measurement uncertainty.
