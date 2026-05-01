# Stock Price Trend Predictor

A comprehensive machine learning project that predicts stock price trends. This repository features traditional Machine Learning models (Linear Regression, tuned Random Forest) as well as Deep Learning (LSTM) for time-series forecasting.

## 📂 Project Structure
The project is organized into a professional Data Science structure:
```text
Stock price trend predictor/
├── data/                    
│   ├── raw/                 - Raw historical data (e.g., aapl_ticker.csv)
│   ├── interim/             - Cleaned data and engineered features
│   └── processed/           - Fully normalized data ready for complex modeling
├── src/                     
│   ├── data_prep/           - Scripts for cleaning, feature engineering (MACD, RSI, etc.)
│   └── modeling/            - Scripts for training and evaluating predictive models
├── requirements.txt         
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites
Install the required dependencies via the included requirements file:
```bash
pip install -r requirements.txt
```

### 2. Workflow & How to Run
All scripts should be executed from the root directory.

1. **Preprocessing & Feature Engineering**: 
   ```bash
   python src/data_prep/preprocess_data.py
   python src/data_prep/featureadd.py
   ```
2. **Traditional ML Models**: 
   ```bash
   python src/modeling/train_model2.py
   ```
3. **Deep Learning Model (LSTM)**: 
   ```bash
   python src/modeling/lstmmodel.py
   ```

## 📊 Models & Methodology

- **Linear Regression**: Used as a baseline to identify simple linear relationships in historical price data.
- **Random Forest Regressor**: An ensemble method to capture non-linear trends. The model is tuned with advanced momentum features like RSI, MACD, and Bollinger Bands to optimize performance.
- **LSTM (Long Short-Term Memory)**: A specialized Recurrent Neural Network (RNN) designed for time-series data. It looks back at a 60-day window to learn complex, sequential price patterns and internal momentum, achieving a sophisticated understanding of stock volatility.

## 📈 Results & Evaluation

The models are evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R²), and Direction Accuracy. 

The predictive outputs provide visual graphs comparing the models' predicted price trajectories against actual historical closing prices.

<img width="920" height="255" alt="image" src="https://github.com/user-attachments/assets/55b586cd-e233-4d51-9a40-f9c91103cf61" />
*(Above: Example results of the initial baseline models on feature-engineered datasets)*
