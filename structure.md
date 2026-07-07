# Project Architecture
```
WindPowerAnalytics/

│
├── config/
│      config.py
│
├── datasets/
│      jogimatti.csv
│      vvs.csv
│
├── preprocessing/
│      preprocess.py
│      feature_engineering.py
│      power_curve.py
│      sequence_generator.py
│      weibull.py
│
├── models/
│      cnn_bilstm.py
│      lightgbm_bilstm.py
│      attention.py
│      losses.py
│
├── training/
│      trainer.py
│      train_cnn.py
│      train_lightgbm.py
│
├── evaluation/
│      metrics.py
│      plots.py
│      uncertainty.py
│      explainability.py
│
├── visualization/
│      dashboard.py
│
├── report/
│      generate_report.py
│
├── outputs/
│
└── README.md
```
### The preprocessing module will perform:
```
CSV
 ↓
Read Dataset

 ↓
Missing Value Handling

 ↓
Outlier Removal (IQR)

 ↓
Feature Engineering

 ↓
Air Density

 ↓
Wind Power

 ↓
ASCTF

 ↓
Lag Generation

 ↓
Sequence Creation

 ↓
Train/Test Split

 ↓
Tensor Creation
```
## CNN-LSTM Architecture
```
Sequence

↓

Conv1D

↓

BatchNorm

↓

ReLU

↓

Conv1D

↓

MaxPool

↓

LSTM

↓

Dropout

↓

Dense

↓

Predicted Wind Speed

↓

ASCTF

↓

Predicted Power
```
## GradientBoost-LSTM
```
Features

↓

Gradient Boosting

↓

Residual Learning

↓

LSTM

↓

Predicted Speed

↓

ASCTF

↓

Power
```
## Evaluation
Every experiment will compute

✓ MAE

✓ RMSE

✓ MAPE

✓ R²

✓ Explained Variance

✓ Pearson Correlation

✓ Maximum Error

✓ Median AE

✓ Normalized RMSE

## Visualizations
Every experiment automatically produces

- Wind speed prediction
- Wind power prediction
- Residual plot
- Error histogram
- Training loss
- Validation loss
- Scatter (Actual vs Predicted)
- QQ plot
- SHAP importance (Gradient Boosting)
- CNN feature maps
- 200-hour telemetry
- Model comparison
## Proposed Research Title
``` Intelligent Hybrid Deep Learning Framework for Short-Term Wind Power Forecasting Using Multi-Scale CNN, BiLSTM, LightGBM and Aerodynamic Soft-Cutoff Transition Filtering```
## Final architecture 
```
                  Raw Weather Data
                          │
──────────────────────────────────────────────────────
          Intelligent Preprocessing Layer
──────────────────────────────────────────────────────
                          │
              Missing Value Handling
                          │
                IQR Outlier Capping
                          │
            Feature Engineering (35+)
                          │
        Dynamic Air Density Computation
                          │
       Weibull Distribution Characterization
                          │
         Sliding Window Generation (6 hr)
                          │
──────────────────────────────────────────────────────
              Model 1                Model 2
──────────────────────────────────────────────────────
 Multi-Scale CNN + BiLSTM       LightGBM + BiLSTM
          │                           │
     Attention Layer            SHAP Feature Selection
          │                           │
   Predicted Wind Speed       Predicted Wind Speed
              \                    /
               \                  /
                ──────────────────
                     ASCTF
                Soft Power Curve
                       │
             Predicted Wind Power
                       │
──────────────────────────────────────────
Evaluation & Explainability Layer
──────────────────────────────────────────
       Metrics
       SHAP
       Uncertainty
       Error Analysis
       Weibull Validation
       PDF Report
```
