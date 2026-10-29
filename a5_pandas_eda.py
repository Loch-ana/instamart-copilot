"""Part A, step 6: pandas cross-check of Q2 and basic exploratory statistics."""
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot table with totals - the grand total must equal the SQL total
pv = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, fill_value=0).round(2)
print(pv.to_string())
print("\nGrand total (pivot):", pv.loc["All", "All"], "  SQL total: 2998668.8")

# per-city sum, mean and count
city = done.groupby("city").agg(total=("order_amount", "sum"),
                                mean=("order_amount", "mean"),
                                count=("order_amount", "count")).round(2)
print("\nPer city:")
print(city.to_string())

# (b) distribution statistics of completed order_amount
x = done["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nOrder amount statistics (completed orders)")
print(f"mean {x.mean():.2f}   median {x.median():.2f}   skewness {x.skew():.2f}")
print(f"Q1 {q1:.2f}   Q3 {q3:.2f}   IQR {iqr:.2f}")
print(f"lower fence {low:.2f}   upper fence {high:.2f}")
print(f"outliers below: {int((x < low).sum())}   outliers above: {int((x > high).sum())}")

corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:")
print(corr.round(3).to_string())
# look at each pair once (ignore the diagonal, which is always 1)
cols = list(corr.columns)
best_pair, best_val = None, 0
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if abs(corr.iloc[i, j]) > abs(best_val):
            best_pair, best_val = (cols[i], cols[j]), corr.iloc[i, j]
print("Largest correlation in absolute value:", best_pair, round(best_val, 3))
