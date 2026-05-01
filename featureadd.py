import pandas as pd
import numpy as np

df = pd.read_csv("preprocessed_data.csv")

print("Original Shape:", df.shape)

# -----------------------------
# 1. EMA (Exponential Moving Average)
# -----------------------------
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# -----------------------------
# 2. RSI (Relative Strength Index)
# -----------------------------
window = 14

delta = df['Close'].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# 3. Volatility (Standard Deviation)
# -----------------------------
df['Volatility_10'] = df['Close'].rolling(window=10).std()

# -----------------------------
# 4. Price Range (Daily Spread)
# -----------------------------
df['High_Low_Range'] = df['High'] - df['Low']

# -----------------------------
# 5. Percentage Change (Momentum)
# -----------------------------
df['Pct_Change'] = df['Close'].pct_change()

# -----------------------------
# 6. MACD (NEW 🔥)
# -----------------------------
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# -----------------------------
# 7. Bollinger Bands (NEW 🔥)
# -----------------------------
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()

df['BB_Middle'] = rolling_mean
df['BB_Upper'] = rolling_mean + (2 * rolling_std)
df['BB_Lower'] = rolling_mean - (2 * rolling_std)

# -----------------------------
# 8. Drop NaN values
# -----------------------------
df = df.dropna()

print("After Feature Engineering Shape:", df.shape)

# -----------------------------
# 9. Save new dataset
# -----------------------------
df.to_csv("final_featured_data.csv", index=False)

print("✅ Feature engineered dataset saved as final_featured_data.csv")