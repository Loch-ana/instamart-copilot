"""Part B, tasks 1-3: RFM table, K-means, segment profile and names, PCA, customer_segments.csv"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ---------------------------------------------------------------- task 1: RFM
snapshot = pd.Timestamp("2025-10-01 00:00:00")
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (snapshot - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
}).reset_index()
print("=== TASK 1: RFM ===")
print("customers:", len(rfm), "(brief 1175)")
print("recency  median", rfm["recency"].median(), " min", rfm["recency"].min(), " max", rfm["recency"].max(), "(19 / 0 / 180)")
print("frequency mean", round(rfm["frequency"].mean(), 2), " max", rfm["frequency"].max(), "(9.46 / 33)")
print("monetary  mean", round(rfm["monetary"].mean(), 2), " max", round(rfm["monetary"].max(), 2), "(2552.06 / 14766.55)")

# ---------------------------------------------------------------- task 2: K-means
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== TASK 2: K-means, K from 2 to 8 ===")
print(f"{'K':>2} {'inertia':>10} {'silhouette':>11}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k:>2} {km.inertia_:10.1f} {silhouette_score(X, km.labels_):11.3f}")
print("(brief: K=4 -> 743.1 / 0.526)")

K = 4   # chosen: clear elbow after 3, best silhouette, and 4 groups the business can act on
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---------------------------------------------------------------- task 3: profile and names
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"),
                                     mean_monetary=("monetary", "mean")).round(2)
print("\n=== TASK 3: segment profile (cluster numbers are only labels, they have no order) ===")
print(prof.to_string())

# names are given from the numbers, so they stay correct even if cluster numbers change order
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest = rest.drop(index=loyal_id)
dormant_id = rest["mean_recency"].idxmax()
names = {premium_id: "Premium High-Spenders", loyal_id: "Loyal Regulars", dormant_id: "At-Risk Dormant"}
rfm["segment_name"] = rfm["segment_id"].map(names).fillna("Occasional Buyers")
rfm["is_premium"] = rfm["segment_id"] == premium_id

actions = {"Premium High-Spenders": "VIP offers, early access to electronics deals, priority support",
           "Loyal Regulars": "Loyalty points and basket-size bundles to raise spend per order",
           "At-Risk Dormant": "Win-back coupon and a reminder message before they leave for good",
           "Occasional Buyers": "Light nudges and category recommendations to raise order frequency"}
print("\nSegment names and recommended action:")
for sid, row in prof.iterrows():
    nm = rfm.loc[rfm["segment_id"] == sid, "segment_name"].iloc[0]
    print(f"  segment {sid}: {nm} (size {int(row['size'])}) -> {actions[nm]}")

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print(f"\nPCA explained variance: {r[0]:.3f} and {r[1]:.3f}, total {r.sum():.3f} (brief 0.634 / 0.218 / 0.852)")

rfm["monetary"] = rfm["monetary"].round(2)
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
print("Saved outputs/customer_segments.csv")
