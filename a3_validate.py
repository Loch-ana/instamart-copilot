"""Part A, step 4: check the clean data against the numbers given in the brief."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o["status"] == "completed"]
cancel = o[o["status"] == "cancelled"]
cities = c["city"].value_counts().to_dict()
genders = c["gender"].value_counts().to_dict()

con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
db_rows = con.execute("select count(*) from orders").fetchone()[0]
con.close()

checks = {
    "12,309 order rows": len(o) == 12309,
    "one row per order_id": o["order_id"].is_unique,
    "11,110 completed orders": len(done) == 11110,
    "1,199 cancelled orders": len(cancel) == 1199,
    "no pending rows": (o["status"] == "pending").sum() == 0,
    "0 missing order_amount": o["order_amount"].isna().sum() == 0,
    "0 missing delivery (completed)": done["delivery_minutes"].isna().sum() == 0,
    "1,199 missing delivery (cancelled)": cancel["delivery_minutes"].isna().sum() == 1199,
    "completed revenue = 2,998,668.80": round(done["order_amount"].sum(), 2) == 2998668.80,
    "six cities with correct counts": cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                                  "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts 610/560/30": genders == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages": c["age"].isna().sum() == 0,
    "ages between 18 and 60": c["age"].between(18, 60).all(),
    "database has 3 tables": {"customers", "products", "orders"} <= tables,
    "database orders = 12,309 rows": db_rows == 12309,
}
ok = True
for name, passed in checks.items():
    print(("PASS  " if passed else "FAIL  ") + name)
    ok = ok and bool(passed)
assert ok, "Some checkpoints failed - do not continue until they pass"
print("ALL CHECKPOINTS PASSED")
