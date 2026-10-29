"""Part B, tasks 4-8: lapse problem, preprocessing, tuning, balance experiment, interpretation, saving."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
seg = pd.read_csv("outputs/customer_segments.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# ---------------------------------------------------------------- task 4: define the problem
CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
before = orders[orders["order_ts"] < CUTOFF]            # ONLY orders before the cutoff make features
done_before = before[before["status"] == "completed"]
done_sep = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]

g = done_before.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().index[0]),
}).reset_index()
final = before[before["status"].isin(["completed", "cancelled"])]
cancel_rate = (final["status"] == "cancelled").groupby(final["customer_id"]).mean().rename("cancel_rate").reset_index()
feat = feat.merge(cancel_rate, on="customer_id", how="left").fillna({"cancel_rate": 0})
feat = feat.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat["customer_id"].isin(done_sep["customer_id"])).astype(int)
feat = feat.drop(columns=["signup_date"])

print("=== TASK 4: lapse problem ===")
print("cohort:", len(feat), "(brief 1156)   lapsed:", int(feat["lapsed"].sum()),
      f"({feat['lapsed'].mean() * 100:.2f}%)   (brief 443, 38.32%)")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train rows:", len(X_train), " test rows:", len(X_test), "(brief 924 / 232)")
base_pred = np.zeros(len(y_test), dtype=int)
print("BASELINE (always 'not lapsed'): accuracy", round(accuracy_score(y_test, base_pred) * 100, 1),
      "%  F1 of lapsed", f1_score(y_test, base_pred, zero_division=0), "(brief 61.6% / 0)")

# ---------------------------------------------------------------- task 5: preprocessing without leakage
print("\n=== TASK 5: preprocessing ===")
print("missing values in any feature column:", int(X.isna().sum().sum()))
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
pre_check = make_pre().fit(X_train)                       # fitted on TRAINING rows only
learned = pre_check.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"scaler mean of frequency (train rows only): {learned:.3f}   full-cohort mean: {feat['frequency'].mean():.3f}")
print("They differ because the scaler only saw the 924 training rows, never the test rows, so it cannot use test information.")

# ---------------------------------------------------------------- task 6: tune and evaluate
print("\n=== TASK 6: tuning (this takes a minute) ===")
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print("Logistic regression best:", lr_gs.best_params_, " CV F1", round(lr_gs.best_score_, 3))
print("Random forest best:      ", rf_gs.best_params_, " CV F1", round(rf_gs.best_score_, 3))

def report(name, model, Xt, yt):
    p = model.predict(Xt)
    print(f"\n{name}")
    print("  confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(yt, p).tolist())
    print(f"  lapsed class: precision {precision_score(yt, p, zero_division=0):.3f}  "
          f"recall {recall_score(yt, p, zero_division=0):.3f}  F1 {f1_score(yt, p, zero_division=0):.3f}")
    return f1_score(yt, p, zero_division=0)

print("\nTEST SET (used only now, after tuning)")
f1_lr = report("Logistic regression", lr_gs.best_estimator_, X_test, y_test)
f1_rf = report("Random forest", rf_gs.best_estimator_, X_test, y_test)
best_is_rf = f1_rf >= f1_lr
best_name = "Random forest" if best_is_rf else "Logistic regression"
best_model = rf_gs.best_estimator_ if best_is_rf else lr_gs.best_estimator_
print(f"\nSELECTED MODEL: {best_name} (higher test F1; must be at least 0.55 -> {max(f1_lr, f1_rf) >= 0.55})")
print("Why: missing a customer who is about to lapse (a false negative) loses future revenue, while wrongly "
      "sending a coupon to a loyal customer (a false positive) costs only a small discount, so we prefer the "
      "model that catches more lapsing customers at a good balance, which the lapsed-class F1 measures.")

# ---------------------------------------------------------------- task 7: class-balance experiment
print("\n=== TASK 7: over-sampling the training set only ===")
Xtr_reset, ytr_reset = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
n_major, n_minor = int((ytr_reset == 0).sum()), int((ytr_reset == 1).sum())
extra = ytr_reset[ytr_reset == 1].sample(n=n_major - n_minor, replace=True, random_state=42).index
Xbal = pd.concat([Xtr_reset, Xtr_reset.loc[extra]], ignore_index=True)
ybal = pd.concat([ytr_reset, ytr_reset.loc[extra]], ignore_index=True)
print("balanced training counts:", ybal.value_counts().to_dict(), "(brief 570 each)")
lr_best_C = lr_gs.best_params_["clf__C"]
lr_bal = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, C=lr_best_C))]).fit(Xbal, ybal)
print("--- before balancing ---"); report("Logistic regression (original)", lr_gs.best_estimator_, X_test, y_test)
print("--- after balancing ---");  report("Logistic regression (balanced)", lr_bal, X_test, y_test)
print("Recall moves most (it rises), precision stays about the same: the retention team finds many more "
      "lapsing customers but must accept that about as many of those contacted are false alarms.")

# ---------------------------------------------------------------- task 8: interpret, save, use
print("\n=== TASK 8: interpretation ===")
rf_final = rf_gs.best_estimator_
names = rf_final.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_final.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("Top 5 random-forest features:")
print(imp.head(5).round(3).to_string())
lr_final = lr_gs.best_estimator_
coef = pd.Series(lr_final.named_steps["clf"].coef_[0], index=lr_final.named_steps["pre"].get_feature_names_out())
print("\nLogistic regression coefficients (positive = more likely to lapse):")
for f in ["num__recency", "num__frequency", "num__monetary"]:
    print(f"  {f}: {coef[f]:+.3f}")

joblib.dump(best_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = bool((reloaded.predict(X_test) == best_model.predict(X_test)).all())
print("\nSaved outputs/lapse_model.joblib; reloaded model gives identical test predictions:", same)
assert same

proba = best_model.predict_proba(X_test)[:, 1]
res = pd.DataFrame({"customer_id": feat.loc[X_test.index, "customer_id"].values,
                    "lapse_probability": proba.round(4), "actually_lapsed": y_test.values})
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
ret = res[res["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print("\nRetention list (Premium customers in the test set):")
print(ret.to_string(index=False))
print("Of these", len(ret), ",", int(ret["actually_lapsed"].sum()), "truly lapsed.")
print("NOTE: the Premium flag uses whole-window RFM (it includes September), so this list is illustrative "
      "and not a leakage-free evaluation.")
