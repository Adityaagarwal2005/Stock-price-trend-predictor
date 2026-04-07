import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Step 2: Define features
features = ['lag1', 'lag2', 'lag3', 'SMA_10', 'SMA_50', 'returns', 'Volume']

X = df[features]
y = df['Close']

# Step 3: Train-test split (time-based)
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Step 4: Train models

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Step 6: Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("----- Linear Regression -----")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

print("\n----- Random Forest -----")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))




plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(rf_pred, label="Predicted (RF)")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()