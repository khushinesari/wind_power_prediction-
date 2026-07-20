#  Hybrid Machine Learning Framework for Short-Term Wind Speed and Power Forecasting

An end-to-end hybrid machine learning and deep learning framework for **short-term wind speed forecasting** and **physics-aware wind power estimation** using statistical preprocessing, temporal feature engineering, and advanced forecasting models.

The project integrates probabilistic data preprocessing, hybrid forecasting architectures, and aerodynamic power conversion to provide an accurate and scalable solution for renewable energy forecasting.

---

## 📌 Overview

Wind energy is inherently stochastic, making accurate short-term forecasting essential for efficient grid integration and turbine operation. This project presents a unified forecasting pipeline that combines statistical preprocessing, machine learning, and deep learning techniques to improve wind speed prediction while estimating wind power using an aerodynamic conversion model.

The framework includes:

- Data preprocessing and cleaning
- Gamma distribution-based outlier handling
- Temporal feature engineering
- Wind vector transformation
- Hybrid machine learning and deep learning models
- Physics-aware wind speed-to-power conversion
- Automated evaluation and report generation

---

## 🚀 Features

- Statistical preprocessing using Gamma distribution
- Missing value handling and outlier removal
- Temporal feature engineering
- 2D Cartesian wind vector representation
- Chronological train-validation-test split
- Multiple forecasting models
- Physics-aware wind power estimation
- Automated experiment evaluation
- Visualization and report generation

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib
- OpenPyXL

---

## 📂 Project Structure

```text
WindForecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── cnn_bilstm.py
│   ├── gb_bilstm.py
│   ├── patchtst.py
│
├── preprocessing/
│   ├── preprocess.py
│   ├── feature_engineering.py
│
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   ├── power_conversion.py
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   ├── predictions/
│
├── train_location.py
├── generate_report.py
├── main.py
└── README.md
```

---

## 📊 Workflow

```text
Raw Wind Farm Data
        │
        ▼
Data Cleaning
        │
        ▼
Gamma Distribution-Based Filtering
        │
        ▼
Feature Engineering
        │
        ▼
Wind Direction → Cartesian Components
        │
        ▼
Sequence Generation
        │
        ▼
Model Training
(CNN-BiLSTM / GB-BiLSTM / PatchTST)
        │
        ▼
Wind Speed Prediction
        │
        ▼
Physics-Based Power Conversion
        │
        ▼
Performance Evaluation
        │
        ▼
Visualization & Report Generation
```

---

## 📈 Models Implemented

The project evaluates multiple forecasting architectures including:

- CNN-BiLSTM
- Gradient Boosting + BiLSTM
- PatchTST (Transformer)
- Random Forest
- XGBoost
- LightGBM
- Multi-Layer Perceptron (MLP)

---

## ⚙️ Feature Engineering

The preprocessing pipeline generates several informative features:

### Temporal Features

- Hour
- Day
- Month
- Season
- Rolling Mean
- Rolling Standard Deviation

### Meteorological Features

- Wind Speed
- Wind Direction
- Temperature
- Pressure
- Humidity
- Air Density

### Derived Features

- Wind Vector (U Component)
- Wind Vector (V Component)
- Lag Features
- Moving Statistics

---

## 🌪️ Statistical Preprocessing

The framework employs Gamma distribution-based statistical preprocessing to improve data quality.

The preprocessing pipeline includes:

- Missing value handling
- Invalid measurement removal
- Gamma distribution fitting
- Probabilistic outlier filtering
- Feature normalization

---

## ⚡ Physics-Aware Wind Power Estimation

Instead of directly predicting turbine power, the framework first forecasts wind speed and subsequently estimates wind power using an aerodynamic power conversion model.

The power estimation considers:

- Air density
- Rotor swept area
- Power coefficient
- Cut-in and cut-out characteristics
- Soft transition near operational limits

This hybrid strategy preserves physical consistency while leveraging data-driven forecasting.

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Coefficient of Determination (R²)

---

## 📈 Outputs

The framework automatically generates:

- Wind speed prediction plots
- Wind power prediction plots
- Training curves
- Error analysis
- Excel result sheets
- Performance reports
- Publication-ready figures

---

## 📁 Dataset

The framework is designed for structured wind farm meteorological datasets containing variables such as:

- Wind Speed
- Wind Direction
- Temperature
- Humidity
- Atmospheric Pressure
- Air Density

The preprocessing pipeline can be adapted to other wind farm datasets with similar features.

---

## 🔬 Applications

- Renewable Energy Forecasting
- Smart Grids
- Wind Farm Monitoring
- Energy Management Systems
- Time Series Forecasting
- AI for Sustainable Energy

---

## 📌 Future Work

- Probabilistic forecasting
- Multi-horizon prediction
- Explainable AI (XAI)
- Uncertainty quantification
- Transformer-based ensemble learning
- Real-time deployment
- Digital twin integration

---

## 🤝 Contributions

Contributions, feature requests, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

## 👩‍💻 Author

**Khushi N**

B.Tech Electronics and Communication Engineering  
PES University

**Interests**

- Artificial Intelligence
- Machine Learning
- Time Series Forecasting
- Renewable Energy Analytics
- Computer Vision
