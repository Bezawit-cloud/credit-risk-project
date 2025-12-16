# task4_proxy_target.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ------------------------
# 1️⃣ Load Processed WOE Features and Original Data
# ------------------------
features_path = (
    r"C:\Users\bezis\Downloads\credit-risk-project\data\processed\features_woe.csv"
)
raw_data_path = (
    r"C:\Users\bezis\Downloads\credit-risk-project\data\data.csv"
)

df_features = pd.read_csv(features_path)
df_raw = pd.read_csv(raw_data_path)

# ------------------------
# 2️⃣ Calculate RFM Metrics
# ------------------------
print("Calculating RFM metrics...")

# Ensure TransactionStartTime is datetime
df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'])

# Snapshot date (one day after the latest transaction)
snapshot_date = df_raw['TransactionStartTime'].max() + pd.Timedelta(days=1)

# Aggregate per CustomerId
rfm = df_raw.groupby('CustomerId').agg(
    recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    frequency=('TransactionId', 'count'),
    monetary=('Amount', 'sum')
).reset_index()

# ------------------------
# 3️⃣ Scale RFM Features for Clustering
# ------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# ------------------------
# 4️⃣ Cluster Customers using K-Means
# ------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# ------------------------
# 5️⃣ Identify High-Risk Cluster
# ------------------------
# High-risk: High recency (inactive), low frequency, low monetary
cluster_stats = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
# Sort by recency descending, frequency ascending, monetary ascending
cluster_stats = cluster_stats.sort_values(
    by=['recency', 'frequency', 'monetary'],
    ascending=[False, True, True]
)
high_risk_cluster = cluster_stats.index[0]

# Assign binary target
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# ------------------------
# 6️⃣ Merge Target with WOE Features
# ------------------------
df_final = df_features.merge(
    rfm[['CustomerId', 'is_high_risk']],
    on='CustomerId',
    how='left'
)

# ------------------------
# 7️⃣ Save Final Dataset
# ------------------------
output_folder = (
    r"C:\Users\bezis\Downloads\credit-risk-project\data\processed"
)
os.makedirs(output_folder, exist_ok=True)
final_path = os.path.join(output_folder, "features_woe_with_target.csv")
df_final.to_csv(final_path, index=False)

print("Task 4 completed!")
print("Final dataset saved at:", final_path)
print("Sample of target column:")
print(df_final[['CustomerId', 'is_high_risk']].head())
