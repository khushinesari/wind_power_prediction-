"""
model_gb_lstm.py
==================
Hybrid Gradient Boosting (XGBoost) + LSTM stacking ensemble (Strategy C /
indirect: predicts wind speed, converted to power via Gamma-ASCTF).

Design rationale:
  - XGBoost operates on the flattened 6-hour lag window (tabular view) and
    excels at capturing non-linear feature interactions (e.g. humidity x
    pressure x lagged speed) without needing large data volumes.
  - An LSTM operates on the same window in true sequence form and captures
    temporal momentum/trend that a tree ensemble treats as independent
    columns.
  - A lightweight Ridge meta-learner stacks the two out-of-fold prediction
    streams into a single final wind-speed forecast, so the ensemble can
    lean on whichever base learner is locally more reliable.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import xgboost as xgb


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=lstm_hidden,
                             num_layers=lstm_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def _train_lstm(X_train, y_train, X_val, y_val, epochs=30, lr=1e-3,
                 batch_size=64, peak_weight=1.5, device="cpu", verbose=True):
    from wind_common import weighted_mse_loss

    model = LSTMForecaster(n_features=X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    n = Xtr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_mse_loss(pred, yb, peak_weight=peak_weight)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                val_loss = weighted_mse_loss(model(Xva), yva, peak_weight=peak_weight).item()
            print(f"  [GB+LSTM: LSTM leg] epoch {epoch+1:02d}/{epochs}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")
    return model


def train_gb_lstm_hybrid(X_train_seq, y_train, X_val_seq, y_val,
                          xgb_params=None, epochs=30, device="cpu", verbose=True):
    """
    Returns a dict bundle {xgb_model, lstm_model, meta_model} implementing
    the stacking hybrid. X_*_seq are (N, lag, n_features) sequence arrays;
    XGBoost consumes a flattened view internally.
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=1,
        )

    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)

    # --- Base learner 1: XGBoost on flattened lag window ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_flat, y_train)

    # --- Base learner 2: LSTM on true sequence view ---
    lstm_model = _train_lstm(X_train_seq, y_train, X_val_seq, y_val,
                              epochs=epochs, device=device, verbose=verbose)

    # --- Out-of-fold predictions on the training set for honest stacking ---
    kf = KFold(n_splits=5, shuffle=False)
    oof_xgb = np.zeros(len(y_train))
    oof_lstm = np.zeros(len(y_train))
    for tr_idx, ho_idx in kf.split(X_train_flat):
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_train_flat[tr_idx], y_train[tr_idx])
        oof_xgb[ho_idx] = m_xgb.predict(X_train_flat[ho_idx])

        m_lstm = _train_lstm(X_train_seq[tr_idx], y_train[tr_idx],
                              X_train_seq[ho_idx], y_train[ho_idx],
                              epochs=max(10, epochs // 2), device=device, verbose=False)
        with torch.no_grad():
            xt = torch.tensor(X_train_seq[ho_idx], dtype=torch.float32, device=device)
            oof_lstm[ho_idx] = m_lstm(xt).cpu().numpy()

    meta_model = Ridge(alpha=1.0, positive=True)
    meta_model.fit(np.column_stack([oof_xgb, oof_lstm]), y_train)

    if verbose:
        print(f"  [GB+LSTM] meta-learner weights (xgb, lstm): {meta_model.coef_}, intercept={meta_model.intercept_:.4f}")

    return {"xgb": xgb_model, "lstm": lstm_model, "meta": meta_model}


def predict_gb_lstm_hybrid(bundle, X_seq, device="cpu"):
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    pred_xgb = bundle["xgb"].predict(X_flat)
    with torch.no_grad():
        xt = torch.tensor(X_seq, dtype=torch.float32, device=device)
        pred_lstm = bundle["lstm"](xt).cpu().numpy()
    stacked = np.column_stack([pred_xgb, pred_lstm])
    return bundle["meta"].predict(stacked)
