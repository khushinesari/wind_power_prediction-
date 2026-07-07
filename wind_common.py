"""
wind_common.py
==================
Shared utilities for the hybrid wind-power forecasting pipeline.

Follows the preprocessing methodology of the reference report
(Wind_Power_Prediction_Methodology___Strategy_Comparison_Report.pdf):
    Step 1 - Chronological 80/20 split
    Step 2 - IQR outlier capping (no row dropping, sequence continuity preserved)
    Step 3 - Air-density based effective wind speed correction
    Step 4 - Cyclic time encodings + lag sequence construction (L = 6 hours)
    Step 5 - Asymmetric (peak-weighted) loss for PyTorch models

NEW in this pipeline (per user request):
    - A Gamma-distribution based Adaptive Soft-Cutoff Transition Filter
      (G-ASCTF), replacing the original report's symmetric Sigmoid ASCTF
      with a right-skewed Gamma CDF transition -- consistent with the fact
      that wind speed itself is conventionally modeled with a skewed
      distribution (Weibull/Gamma family), not a symmetric one.
    - Ground-truth power is not present in the raw CSVs (only weather +
      wind-speed columns are provided), so a physically-grounded target
      power series is synthesized from wind speed + air density using the
      standard cubic turbine power curve, smoothed with the G-ASCTF.
      This synthesized series is the regression target ("power_mw") for
      every model in this pipeline.
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

# ----------------------------------------------------------------------
# Turbine specification (2.0 MW utility profile, per reference report)
# ----------------------------------------------------------------------
RATED_POWER_MW = 2.0
CUT_IN_MS      = 3.0
RATED_SPEED_MS = 12.5   # assumption: typical rated speed for a 2.0 MW turbine
CUT_OUT_MS     = 25.0
RHO_STD        = 1.225  # kg/m^3, IEC standard air density

LAG_HOURS = 6  # sequence window length, matches reference report


# ----------------------------------------------------------------------
# Step 2: IQR capping (applied to wind speed, temperature, humidity)
# ----------------------------------------------------------------------
def iqr_cap(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


# ----------------------------------------------------------------------
# Step 3: Air-density corrected "effective" wind speed
# P is proportional to rho * v^3, so v_eff = v * (rho / rho_std)^(1/3)
# lets us reuse a single density-standardized power curve.
# ----------------------------------------------------------------------
def effective_wind_speed(v_ms: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return v_ms * np.power(rho / RHO_STD, 1.0 / 3.0)


# ----------------------------------------------------------------------
# Gamma-based Adaptive Soft-Cutoff Transition Filter (G-ASCTF)
# ----------------------------------------------------------------------
def _gamma_rise(x, width, skew_a=2.2):
    """Smooth 0->1 rise using a right-skewed Gamma CDF, transition ~ `width` m/s."""
    scale = width / skew_a
    return gamma_dist.cdf(np.clip(x, 0, None), a=skew_a, scale=scale)


def _gamma_fall(x, width, skew_a=2.6):
    """Smooth 1->0 fall (mirrors the rise using a Gamma survival function)."""
    scale = width / skew_a
    return gamma_dist.sf(np.clip(x, 0, None), a=skew_a, scale=scale)


def gamma_asctf_power(v_ms: np.ndarray, rho: np.ndarray,
                       rated_power=RATED_POWER_MW, cut_in=CUT_IN_MS,
                       rated_speed=RATED_SPEED_MS, cut_out=CUT_OUT_MS) -> np.ndarray:
    """
    Converts wind speed (+ air density) to turbine power output using a
    cubic power curve between cut-in and rated speed, saturating at
    rated power, and softened at both the cut-in and cut-out boundaries
    with a Gamma-CDF transition instead of a hard step / symmetric sigmoid.
    """
    v_ms = np.asarray(v_ms, dtype=float)
    rho = np.asarray(rho, dtype=float)
    v_eff = effective_wind_speed(v_ms, rho)

    # Base cubic curve (clipped at rated power), defined smoothly over all v
    frac = np.clip((v_eff ** 3 - cut_in ** 3) / (rated_speed ** 3 - cut_in ** 3), 0, 1)
    base_power = rated_power * frac

    # Gamma rise centered on cut_in (transition window = 2.0 m/s)
    rise = _gamma_rise(v_eff - (cut_in - 1.0), width=2.0)
    # Gamma fall centered on cut_out (transition window = 3.0 m/s)
    fall = _gamma_fall(v_eff - (cut_out - 1.5), width=3.0)

    power = base_power * rise * fall
    return np.clip(power, 0, rated_power)


# ----------------------------------------------------------------------
# Full preprocessing pipeline for one location CSV
# ----------------------------------------------------------------------
FEATURE_COLS = [
    "temperature_2m_C", "relative_humidity_2m_pct", "surface_pressure_hPa",
    "wind_speed_100m_ms", "air_density_kgm3",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
SPEED_COL = "wind_speed_100m_ms"
TARGET_SPEED_COL = "wind_speed_100m_ms"   # what the indirect models forecast
TARGET_POWER_COL = "power_mw"             # the physically synthesized ground truth


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])
    df = df.sort_values("timestamp_ist").reset_index(drop=True)

    # Step 2: IQR capping (no rows dropped -> sequence continuity preserved)
    for col in ["wind_speed_10m_ms", "wind_speed_100m_ms",
                "temperature_2m_C", "relative_humidity_2m_pct"]:
        df[col] = iqr_cap(df[col])

    # Step 4: cyclic time encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp_ist"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp_ist"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["timestamp_ist"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["timestamp_ist"].dt.month / 12)

    # Synthesized ground-truth power via Gamma-ASCTF (this is our label)
    df[TARGET_POWER_COL] = gamma_asctf_power(
        df["wind_speed_100m_ms"].values, df["air_density_kgm3"].values
    )
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.8):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def build_sequences(df: pd.DataFrame, feature_cols, target_col, lag=LAG_HOURS):
    """Sliding window: X shape (N, lag, n_features), y shape (N,)."""
    feats = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    X, y = [], []
    for i in range(lag, len(df)):
        X.append(feats[i - lag:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def build_flat_features(df: pd.DataFrame, feature_cols, target_col, lag=LAG_HOURS):
    """Flattened lag window for tabular models (e.g. Gradient Boosting)."""
    X_seq, y = build_sequences(df, feature_cols, target_col, lag)
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    return X_flat, y


# ----------------------------------------------------------------------
# Step 5: Asymmetric peak-weighted MSE loss (PyTorch)
# ----------------------------------------------------------------------
def weighted_mse_loss(pred, target, peak_weight=1.5):
    target_scaled = (target - target.mean()) / (target.std() + 1e-8)
    weights = 1.0 + peak_weight * torch_relu(target_scaled)
    return (weights * (pred - target) ** 2).mean()


def torch_relu(x):
    import torch
    return torch.clamp(x, min=0)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
