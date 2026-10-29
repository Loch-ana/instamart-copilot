"""Part A, step 1: audit the raw data (nothing is changed here, we only look)."""
import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print("=" * 60)
    print(f"TABLE: {name}   shape = {df.shape}")
    print("-- data types --")
    print(df.dtypes.to_string())
    print("-- missing values --")
    miss = pd.DataFrame({"missing": df.isna().sum(), "percent": (df.isna().mean() * 100).round(2)})
    print(miss.to_string())

print("=" * 60)
print("SPECIFIC AUDIT FIGURES")
# note: pandas reads the text N/A as a missing value, so the missing table above shows 53 for
# order_amount; the other 67 are the word "error". Together they are the 120 unusable amounts.
amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(customers["age"], errors="coerce")
gender_nonblank = customers["gender"].dropna()

figures = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "customers: blank age": int(customers["age"].isna().sum()),
    "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "customers: distinct raw city spellings": int(customers["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(gender_nonblank.nunique()),
    "customers: blank gender": int(customers["gender"].isna().sum()),
}
reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]
print(f"{'figure':50s} {'yours':>8s} {'brief':>8s}")
for (k, v), ref in zip(figures.items(), reference):
    flag = "OK" if v == ref else "<-- DIFFERENT"
    print(f"{k:50s} {v:8d} {ref:8d}  {flag}")
