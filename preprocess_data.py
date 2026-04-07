import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("aapl_ticker.csv")  

print("Original Data:")
print(df.head())

# Step 2: Standardize column names (remove spaces)
df.columns = df.columns.str.strip()

# Step 3: Convert Date column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
else:
    print("⚠️ No 'Date' column found. Check column names.")

# Step 4: Handle missing values
df = df.dropna()

# Step 5: Feature Engineering

# Lag features
df['lag1'] = df['Close'].shift(1)
df['lag2'] = df['Close'].shift(2)
df['lag3'] = df['Close'].shift(3)
df['lag7'] = df['Close'].shift(7)
df['lag10'] = df['Close'].shift(10)
# Moving averages
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Returns
df['returns'] = df['Close'].pct_change()

# Volume (already present, just ensure no issues)
if 'Volume' not in df.columns:
    print("⚠️ Volume column not found")

# Step 6: Drop NA again (due to lag/rolling)
df = df.dropna()

# Step 7: Save preprocessed data
df.to_csv("preprocessed_data.csv")

print("\n✅ Preprocessing done!")
print("New file saved as 'preprocessed_data.csv'")
print(df.head())