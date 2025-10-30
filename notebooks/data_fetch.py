import requests
import json
import os
import pandas as pd
import numpy as np  
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib
import dotenv

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")

base_URL = "https://api.pricempire.com/v4/paid/items/prices"
headers = {"Authorization": f"Bearer {API_KEY}"}

def fetch_prices():   
    params = {
        "sources": "buff163,dmarket,skinport", 
        "currency": "USD",
        "avg": "true",
        "median": "true"
    }
    response = requests.get(base_URL, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

def fetch_price_history():
    history_URL = f"{base_URL}/history"
    params = {
        "app_id": 730,
        "provider_key": "buff163",
        "currency": "USD",
        "from_date": (date.today() - timedelta(days=149)).strftime("%Y-%m-%d"), 
        "to_date": date.today().strftime("%Y-%m-%d") 
    }

    response = requests.get(history_URL, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    return response.json()


data = fetch_price_history()

records = []
price_data = data["data"] 

for name, history in price_data.items():
    for timestamp, price in history.items():
        records.append((name, timestamp, price))

records_np = np.array(records, dtype=object)

np.set_printoptions(linewidth=150, suppress=True)
# print("Fetched historical data (name, timestamp, price):\n")
# print(records_np)

# Convert to DataFrame
df = pd.DataFrame(records_np, columns=["market_hash_name", "timestamp", "price"])
df["timestamp"] = df["timestamp"].astype(int)
df["price"] = df["price"].astype(float)//100
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.sort_values(by="date")

skin_names = df["market_hash_name"].unique().tolist()
skin_names = sorted(skin_names)

output_path = "../data/raw/skin_list.txt"
import os
if not os.path.exists(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for name in skin_names:
            f.write(name + "\n")

# print(f"✅ Saved {len(skin_names)} unique skins to {output_path}")


df["price_usd"] = df["price"] 

unique_skins = ["Autograph Capsule | Counter Logic Gaming | Cologne 2016"]

# plt.figure(figsize=(14, 8))
# for name in unique_skins:
#     subset = df[df["market_hash_name"] == name]
#     plt.plot(subset["date"], subset["price_usd"], label=name)

# plt.title("Skin Price History", fontsize=14)
# plt.xlabel("Date")
# plt.ylabel("Price (USD)")
# plt.legend(
#     fontsize=8,
#     loc='center left',
#     bbox_to_anchor=(1, 0.5),
#     title="Skins"
# )
# plt.grid(True)
# plt.tight_layout(rect=[0, 0, 0.8, 1])
# plt.show()

df["pct_change"] = df.groupby("market_hash_name")["price_usd"].pct_change()
df["rolling_mean"] = df.groupby("market_hash_name")["price_usd"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())
df["deviation_from_mean"] = (df["price_usd"] - df["rolling_mean"]) / df["rolling_mean"]
df["spike_flag"] = (
    (abs(df["pct_change"]) > 0.5)|        
    (abs(df["deviation_from_mean"]) > 0.7)
)
#Searching for the spike of a particular skin
for row in df[df["spike_flag"]].itertuples():
    if row.market_hash_name == "USP-S | Black Lotus (Factory New)":
        print(f"Spike detected for {row.market_hash_name} on {row.date.date()}:")
        print(f"  Price: ${row.price_usd:.2f}, Pct Change: {row.pct_change:.2%}, Deviation from Mean: {row.deviation_from_mean:.2%}")


df["market_hash_name"] = df["market_hash_name"].str.strip()

unique_skins = [
    "AK-47 | Redline (Field-Tested)",
    "SSG 08 | Ghost Crusader (Factory New)",
    "USP-S | Black Lotus (Factory New)",
    "AUG | Arctic Wolf (Factory New)",
    "AK-47 | Safari Mesh (Factory New)",
    "Sticker | Evil Geniuses (Holo) | Stockholm 2021"
]

feature_df = df.groupby("market_hash_name").agg({
    "price_usd": "mean",
    "pct_change": ["mean", "std"],
    "deviation_from_mean": "mean",
    "spike_flag": "sum"
})

feature_df.columns = ['_'.join(col).strip() for col in feature_df.columns.values]

feature_df = feature_df.rename(columns={
    "price_usd_mean": "mean_price_usd",
    "pct_change_mean": "mean_pct_change",
    "pct_change_std": "volatility",
    "deviation_from_mean_mean": "mean_deviation",
    "spike_flag_sum": "spike_count"
})

# Save feature_df to CSV for use in other notebooks
feature_df_path = "../data/feature_df.csv"
feature_df.to_csv(feature_df_path)
print(f"✅ Saved feature_df to {feature_df_path}")
print(f"   Shape: {feature_df.shape}")
print(f"   Features: {list(feature_df.columns)}")

# Save raw historical data for 3-day price capping
raw_history_path = "../data/raw/price_history.csv"
df.to_csv(raw_history_path, index=False)
print(f"✅ Saved raw historical data to {raw_history_path}")

# Calculate 3-day minimum price for each skin
# Sort by date and get the last 3 days of data per skin
df_sorted = df.sort_values(['market_hash_name', 'date'])

# Group by skin and get the last 3 days
min_price_3day = df_sorted.groupby('market_hash_name').tail(3).groupby('market_hash_name')['price_usd'].min()
min_price_3day.name = 'min_price_3day'

# Add to feature_df
feature_df = feature_df.join(min_price_3day)
feature_df['min_price_3day'] = feature_df['min_price_3day'].fillna(feature_df['mean_price_usd'])

# Save updated feature_df with 3-day minimum
feature_df.to_csv(feature_df_path)
print(f"✅ Updated feature_df with 3-day minimum prices")