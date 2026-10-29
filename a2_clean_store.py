"""Part A, steps 2-3: clean customers and orders, save CSV files and the SQLite database."""
import os
import sqlite3
import pandas as pd

os.makedirs("data/clean", exist_ok=True)

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# ------------------------------------------------------------------ CUSTOMERS
print("=== CLEAN CUSTOMERS ===")
# (a) city: remove spaces, title case
before = customers["city"].nunique()
customers["city"] = customers["city"].str.strip().str.title()
print(f"(a) city spellings: {before} -> {customers['city'].nunique()}")

# (b) gender: male / female / unknown
g = customers["gender"].fillna("").str.strip().str.lower()
customers["gender"] = g.map(lambda x: "male" if x in ("m", "male") else "female" if x in ("f", "female") else "unknown")
print("(b) gender counts:", customers["gender"].value_counts().to_dict())

# (c) age: impossible values become missing, then fill with the city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
bad_age = int(((customers["age"] < 10) | (customers["age"] > 100)).sum())
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = float("nan")
missing_before = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
print(f"(c) impossible ages set to missing: {bad_age}; missing ages before fill: {missing_before}; after fill: {int(customers['age'].isna().sum())}")

# (d) signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print("(d) signup_date converted to datetime")
print(customers["city"].value_counts().to_dict())

# ------------------------------------------------------------------ ORDERS
print("\n=== CLEAN ORDERS ===")
print("rows at start:", len(orders))

# rule 1: order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# rule 2: remove exact duplicates
n = len(orders)
orders = orders.drop_duplicates()
print(f"rule 2: exact duplicate rows removed: {n - len(orders)}")

# rule 3: keep only the latest record per order_id
n = len(orders)
orders = orders.sort_values("order_ts").drop_duplicates(subset="order_id", keep="last")
print(f"rule 3: older (pending) records removed: {n - len(orders)}")

# rule 4: amount to numeric, fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
n_missing = int(orders["order_amount"].isna().sum())
orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
calc = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)
print(f"rule 4: unusable amounts filled from product price: {n_missing}")

# rule 5: negative delivery -> missing; fill completed orders with category median
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
neg = int((orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")
completed = orders["status"] == "completed"
# median of each category, computed from completed orders only
cat_median = orders[completed].groupby("category")["delivery_minutes"].transform("median")
filled_before = int(orders.loc[completed, "delivery_minutes"].isna().sum())
orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(cat_median)
print(f"rule 5: negative delivery times set to missing: {neg}")
print(f"        missing delivery times filled for completed orders: {filled_before}")
print(f"        cancelled orders left with missing delivery time: {int(orders.loc[~completed, 'delivery_minutes'].isna().sum())}")

orders = orders.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("\nrows after cleaning:", len(orders))
print(orders["status"].value_counts().to_dict())
print("sum of completed order_amount:", round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2))

# ------------------------------------------------------------------ SAVE
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")   # text, as required
orders_out.to_csv("data/clean/orders.csv", index=False)

cust_db = customers.copy()
cust_db["signup_date"] = cust_db["signup_date"].dt.strftime("%Y-%m-%d")
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
cust_db.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
orders_out.to_sql("orders", con, index=False)
con.close()
print("\nSaved: data/clean/*.csv and data/instamart.db (tables customers, products, orders)")
