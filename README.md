# Stock Price Prediction using Stacked LSTM

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Keras](https://img.shields.io/badge/Keras-API-yellow)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-blueviolet)

A deep learning project that uses Stacked Long Short-Term Memory (LSTM) networks to predict Apple Inc. (AAPL) stock prices based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
This project implements a 3-layer Stacked LSTM model to predict future stock prices using historical closing prices of AAPL stock. The model demonstrates how deep learning can be applied to financial time-series forecasting.

## Key Features
- Data collection from Tiingo API
- Time-series data preprocessing with MinMax scaling
- Sequential dataset creation with 100-time-step windows
- 3-layer Stacked LSTM architecture implementation
- Model evaluation with RMSE metrics
- Visualization of predictions vs actual values

## Technologies Used
- **Python** (Primary programming language)
- **TensorFlow/Keras** (Deep learning framework)
- **Pandas** (Data manipulation and analysis)
- **NumPy** (Numerical computations)
- **Matplotlib/Seaborn** (Data visualization)
- **scikit-learn** (Data preprocessing)

## Dataset
The dataset contains daily AAPL stock prices from May 2015 to May 2020, including:
- Opening price
- Closing price
- High/Low prices
- Volume
- Adjusted close prices

Data was obtained using the Tiingo API and saved to `AAPL.csv`.

## Implementation
1. **Data Preprocessing**:
   - Normalized data using MinMaxScaler (0-1 range)
   - Split into training (65%) and test (35%) sets
   - Created sequential datasets with 100-time-step windows

2. **Model Architecture**:
   ```python
   model = Sequential()
   model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
   model.add(LSTM(50, return_sequences=True))
   model.add(LSTM(50))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error', optimizer='adam')

## Training
```python
# Model training configuration
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,          # Training epochs
    batch_size=64,       # Batch size
    verbose=1
)

# Model compilation
model.compile(
    loss='mean_squared_error',  # MSE loss function
    optimizer='adam'            # Adam optimizer
)
# Evaluation metrics
print(f"Training RMSE: {math.sqrt(mean_squared_error(y_train, train_predict)):.2f}")
print(f"Test RMSE: {math.sqrt(mean_squared_error(y_test, test_predict)):.2f}")

# Output:
# Training RMSE: 140.99
# Test RMSE: 235.72

## How to Use
# Clone the repository
git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook Stock_Price_Prediction_LSTM.ipynb
