"""Part A, step 5: run the 7 SQL queries from sql/queries.sql and print the key results."""
import sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
queries = [q.strip() for q in text.split(";") if "SELECT" in q or "WITH" in q]
assert len(queries) == 7, f"expected 7 queries, found {len(queries)}"
con = sqlite3.connect("data/instamart.db")
R = [pd.read_sql_query(q, con) for q in queries]
con.close()
q1, q2, q3, q4, q5, q6, q7 = R

print("Q1 customers:", len(q1), "(brief: 30)")
print("Q2 rows:", len(q2), "(30); total:", round(q2["revenue"].sum(), 2), "(2998668.80)")
top = q2.iloc[0]
print("   largest cell:", top["city"], "/", top["category"], top["revenue"], "(Bengaluru / Electronics 233909.45)")
print("Q3 products above average price:", len(q3), "(11)")
print("Q4 customers with no completed order:", len(q4), "(25)")
print("Q5 customers above average:", len(q5), "(550); average:", q5["avg_total"].iloc[0], "(2552.06)")
print("Q6 top customer per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("   rows where RANK differs from ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()), "(8)")
print("Q7 daily rows:", len(q7), "(183); last cumulative:", q7["cumulative_revenue"].iloc[-1], "(2998668.80)")
