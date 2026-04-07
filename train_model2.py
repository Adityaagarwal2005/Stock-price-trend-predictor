import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Step 2: Fix leakage in moving averages (IMPORTANT)
df['SMA_10'] = df['Close'].shift(1).rolling(window=10).mean()
df['SMA_50'] = df['Close'].shift(1).rolling(window=50).mean()

# Step 3: Add extra lag feature
df['lag5'] = df['Close'].shift(5)
df['price_change'] = df['Close'] - df['Open']
df['volatility'] = df['High'] - df['Low']
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
# Step 4: Drop NA again
df = df.dropna()

# Step 5: Define features
features = ['lag1', 'lag2', 'lag3', 'lag5', 'SMA_10', 'SMA_50', 'returns', 'Volume']

X = df[features]
y = df['Close']

# Step 6: Train-test split (time-based)
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Step 7: Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 8: Tuned Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1)

rf.fit(X_train, y_train)

# Step 9: Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Step 10: Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("----- Linear Regression -----")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

print("\n----- Random Forest -----")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

# Step 11: Visualization
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(lr_pred, label="Linear Regression")
plt.plot(rf_pred, label="Random Forest")
plt.legend()
plt.title("Actual vs Predicted (Improved Model)")
plt.show()