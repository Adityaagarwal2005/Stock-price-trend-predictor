Stock Price Trend Predictor

A machine learning project that predicts stock price trends using Linear Regression and Random Forest Regressor, including automated hyperparameter tuning to optimize performance.

📂 Project Structure
preprocessed_data.csv: The cleaned dataset ready for model training.
train_model.py: Initial script implementing Linear Regression and a baseline Random Forest Regressor.
train_model2.py: Advanced script focusing on Random Forest optimization using techniques.

🚀 Getting Started
1. Prerequisites
Ensure you have the following Python libraries installed:
bash
pip install pandas numpy scikit-learn matplotlib seaborn
Use code with caution

2. Workflow

Preprocessing: The data has been pre-cleaned and stored in preprocessed_data.csv to ensure consistent training.
Initial Modeling: Run the first file to see a baseline comparison between a simple Linear model and a standard Random Forest.
Optimization: Run the second file to perform hyperparameter tuning (adjusting n_estimators, max_depth, etc.) to find the best possible Random Forest configuration.

📊 Models & Methodology
Linear Regression: Used as a baseline to identify linear relationships in historical price data.
Random Forest Regressor: An ensemble method used to capture complex, non-linear trends.
Hyperparameter Tuning: We optimized the Random Forest by testing various combinations of tree depth and estimator counts to reduce error.

📈 Results & Evaluation
The models are evaluated using metrics such as Mean Squared Error (MSE) and R-squared (
) to determine which predictor most accurately follows the actual stock trend.In last it also give the graph comparing the two models results

<img width="920" height="255" alt="image" src="https://github.com/user-attachments/assets/55b586cd-e233-4d51-9a40-f9c91103cf61" />
above are the result of linear regression and randomforestregressor for the dataset with added features like EMA_10 ,EMA_50 ,Pct_Change ,RSI,Volatility_10,MACD,MACD,Bollinger Bands
