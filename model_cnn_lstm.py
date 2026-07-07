"""
model_cnn_lstm.py
==================
Hybrid CNN + LSTM architecture (Strategy C / indirect: predicts wind speed,
which is then converted to power via the Gamma-ASCTF curve).

Design rationale:
  - 1D convolutional front-end extracts short-range local patterns across
    the 6-hour lag window (gust structure, rate-of-change) independent of
    absolute position in the window.
  - The convolved feature maps are then fed into an LSTM to model the
    longer-range temporal dependency and produce a single next-step
    wind-speed forecast.
  - Trained with the asymmetric (peak-weighted) MSE loss from the
    reference report, applied on wind speed instead of power.
"""

import torch
import torch.nn as nn


class CNNLSTMHybrid(nn.Module):
    def __init__(self, n_features, seq_len, conv_channels=32, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=conv_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden,
                             num_layers=lstm_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> conv expects (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)          # back to (batch, seq_len, conv_channels)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, lstm_hidden)
        return self.head(last_hidden).squeeze(-1)


def train_cnn_lstm(model, X_train, y_train, X_val, y_val, epochs=30, lr=1e-3,
                    batch_size=64, peak_weight=1.5, device="cpu", verbose=True):
    from wind_common import weighted_mse_loss

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    n = Xtr.shape[0]
    history = []
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

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = weighted_mse_loss(val_pred, yva, peak_weight=peak_weight).item()
        history.append((epoch_loss, val_loss))
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  [CNN+LSTM] epoch {epoch+1:02d}/{epochs}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")
    return history


def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(Xt).cpu().numpy()
