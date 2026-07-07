"""
train_location.py
==================
Trains BOTH hybrid architectures (CNN+LSTM and Gradient-Boosting+LSTM) for a
single wind farm location, using the indirect Strategy-C style pipeline:

    lag features --> [hybrid model] --> wind speed forecast
                  --> Gamma-ASCTF curve --> power forecast (MW)

Usage:
    python train_location.py --csv data/jogimatti_wind_data_2021.csv --name Jogimatti
    python train_location.py --csv data/vvs_wind_data_2021.csv --name VVS
"""

import argparse
import json
import time

import numpy as np
import torch

from wind_common import (
    load_and_prepare, chronological_split, build_sequences,
    FEATURE_COLS, TARGET_SPEED_COL, TARGET_POWER_COL,
    gamma_asctf_power, compute_metrics, LAG_HOURS,
)
from model_cnn_lstm import CNNLSTMHybrid, train_cnn_lstm, predict as cnn_lstm_predict
from model_gb_lstm import train_gb_lstm_hybrid, predict_gb_lstm_hybrid

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def standardize(train_arr, *other_arrs):
    """Standardize features using train-set statistics only (no leakage)."""
    mean = train_arr.mean(axis=(0, 1), keepdims=True)
    std = train_arr.std(axis=(0, 1), keepdims=True) + 1e-8
    out = [(train_arr - mean) / std]
    for a in other_arrs:
        out.append((a - mean) / std)
    return out, mean, std


def main(csv_path, location_name, epochs, out_json):
    t0 = time.time()
    print(f"\n===== Location: {location_name} =====")
    df = load_and_prepare(csv_path)
    train_df, test_df = chronological_split(df, train_frac=0.8)
    print(f"Rows: total={len(df)}  train={len(train_df)}  test={len(test_df)}")

    # Build sequences for wind-speed forecasting target
    X_train, y_train_speed = build_sequences(train_df, FEATURE_COLS, TARGET_SPEED_COL, lag=LAG_HOURS)
    X_test, y_test_speed = build_sequences(test_df, FEATURE_COLS, TARGET_SPEED_COL, lag=LAG_HOURS)

    # Ground-truth power for the SAME rows (aligned with y_*_speed indices)
    y_train_power = build_sequences(train_df, FEATURE_COLS, TARGET_POWER_COL, lag=LAG_HOURS)[1]
    y_test_power = build_sequences(test_df, FEATURE_COLS, TARGET_POWER_COL, lag=LAG_HOURS)[1]

    # Air density aligned with the same rows, needed to convert predicted
    # speed -> power via the Gamma-ASCTF curve
    density_train = build_sequences(train_df, FEATURE_COLS, "air_density_kgm3", lag=LAG_HOURS)[1]
    density_test = build_sequences(test_df, FEATURE_COLS, "air_density_kgm3", lag=LAG_HOURS)[1]

    # Standardize inputs (train stats only)
    (X_train_s, X_test_s), _, _ = standardize(X_train, X_test)

    # Simple internal validation split (last 15% of train, still chronological)
    n_tr = X_train_s.shape[0]
    val_cut = int(n_tr * 0.85)
    X_tr, X_val = X_train_s[:val_cut], X_train_s[val_cut:]
    y_tr, y_val = y_train_speed[:val_cut], y_train_speed[val_cut:]

    results = {"location": location_name, "n_train": int(len(train_df)), "n_test": int(len(test_df))}

    # ---------------------------------------------------------------
    # Hybrid 1: CNN + LSTM
    # ---------------------------------------------------------------
    print("\n-- Training CNN+LSTM hybrid (indirect speed forecast) --")
    cnn_lstm = CNNLSTMHybrid(n_features=X_train_s.shape[2], seq_len=LAG_HOURS)
    train_cnn_lstm(cnn_lstm, X_tr, y_tr, X_val, y_val, epochs=epochs)
    speed_pred_cnn = cnn_lstm_predict(cnn_lstm, X_test_s)
    power_pred_cnn = gamma_asctf_power(speed_pred_cnn, density_test)

    metrics_speed_cnn = compute_metrics(y_test_speed, speed_pred_cnn)
    metrics_power_cnn = compute_metrics(y_test_power, power_pred_cnn)
    print(f"  CNN+LSTM  -> speed R2={metrics_speed_cnn['R2']:.4f} | power MAE={metrics_power_cnn['MAE']:.4f} "
          f"RMSE={metrics_power_cnn['RMSE']:.4f} R2={metrics_power_cnn['R2']:.4f}")

    # ---------------------------------------------------------------
    # Hybrid 2: Gradient Boosting + LSTM (stacked)
    # ---------------------------------------------------------------
    print("\n-- Training Gradient-Boosting + LSTM hybrid (stacked, indirect speed forecast) --")
    gb_lstm_bundle = train_gb_lstm_hybrid(X_tr, y_tr, X_val, y_val, epochs=epochs)
    speed_pred_gb = predict_gb_lstm_hybrid(gb_lstm_bundle, X_test_s)
    power_pred_gb = gamma_asctf_power(speed_pred_gb, density_test)

    metrics_speed_gb = compute_metrics(y_test_speed, speed_pred_gb)
    metrics_power_gb = compute_metrics(y_test_power, power_pred_gb)
    print(f"  GB+LSTM   -> speed R2={metrics_speed_gb['R2']:.4f} | power MAE={metrics_power_gb['MAE']:.4f} "
          f"RMSE={metrics_power_gb['RMSE']:.4f} R2={metrics_power_gb['R2']:.4f}")

    results["CNN_LSTM"] = {"speed_metrics": metrics_speed_cnn, "power_metrics": metrics_power_cnn}
    results["GB_LSTM"] = {"speed_metrics": metrics_speed_gb, "power_metrics": metrics_power_gb}

    # Save arrays needed for plotting later
    np.savez(
        f"outputs/{location_name}_predictions.npz",
        y_test_power=y_test_power,
        power_pred_cnn=power_pred_cnn,
        power_pred_gb=power_pred_gb,
        y_test_speed=y_test_speed,
        speed_pred_cnn=speed_pred_cnn,
        speed_pred_gb=speed_pred_gb,
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Results saved to {out_json}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_json = args.out or f"outputs/{args.name}_results.json"
    main(args.csv, args.name, args.epochs, out_json)
