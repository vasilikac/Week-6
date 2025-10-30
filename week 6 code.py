# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 18:51:38 2025

@author: VasilikaC
"""

# ======================== SIMPLE PIPELINE (15 features, portfolio target) ========================
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel

import yfinance as yf
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # optional

# ------------------ Settings ------------------
BASE_PATH = Path(r"C:\Users\VasilikaC\OneDrive - CKH CPA\Desktop\Tsom erg")
OUT = (BASE_PATH / "outputs_simple"); OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ['HCA','ELV','LRCX','GOOGL','TSLA','MC.PA','NVDA','AAPL','MSFT']
START_DATE = '2015-10-28'
END_DATE   = datetime.date.today().strftime('%Y-%m-%d')

SEED = 42
TEST_SIZE = 0.20

# ------------------ 1) Data ------------------
print(f"Λήψη: {', '.join(TICKERS)}  ({START_DATE} → {END_DATE})")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                   interval="1d", auto_adjust=True, progress=False)

# Close prices
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Close'].copy()
else:
    prices = data.copy()

prices = prices.reindex(columns=TICKERS).ffill()
returns = prices.pct_change() * 100.0
returns = returns.dropna(how="any").astype(float)
returns.index = pd.to_datetime(returns.index)
print("returns shape:", returns.shape)

# Save raw (optional)
try:
    with pd.ExcelWriter(BASE_PATH / "data_prices_and_returns.xlsx", engine='openpyxl') as w:
        prices.to_excel(w, sheet_name='prices_adjclose', index=True)
        returns.to_excel(w, sheet_name='daily_returns_%', index=True)
except Exception as e:
    print("Excel save skipped:", e)
returns.to_csv(BASE_PATH / "aligned_df.csv", sep=';', decimal=',')

# ------------------ 2) 15 Features (clean, no leakage) ------------------
# 9 features: lag1 for EACH asset (on returns)
X = pd.DataFrame(index=returns.index)
for t in TICKERS:
    X[f"{t}_lag1"] = (returns[t] / 100.0).shift(1)

# 3 features: market momentum 3/5/10 (equal-weight market)
mkt = (returns[TICKERS] / 100.0).mean(axis=1)
X["mkt_mom3"]  = mkt.rolling(3).sum()
X["mkt_mom5"]  = mkt.rolling(5).sum()
X["mkt_mom10"] = mkt.rolling(10).sum()

# 2 features: market volatility 10/20
X["mkt_vol10"] = mkt.rolling(10).std()
X["mkt_vol20"] = mkt.rolling(20).std()

# 1 feature: AAPL distance from MA20 (price-based)
if "AAPL" not in prices.columns:
    raise ValueError("AAPL λείπει από τα prices—έλεγξε τα tickers.")
ma20 = prices["AAPL"].rolling(20).mean()
X["AAPL_ma20_dist"] = prices["AAPL"] / ma20 - 1.0

# Drop initial NaNs from lags/rollings
X = X.dropna().copy()
print("features (15) shape:", X.shape)

# ------------------ 3) Target = equal-weight portfolio return ------------------
y = returns[TICKERS].mean(axis=1).reindex(X.index).to_numpy().ravel()

# ------------------ 4) Split / Scale ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ------------------ 5) Models & TimeSeries CV ------------------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=SEED),
}
if XGBRegressor is not None:
    models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=SEED)

cv = TimeSeriesSplit(n_splits=5)

rows, abs_err = [], {}
for name, model in models.items():
    cv_r2  = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="r2").mean()
    cv_mse = -cross_val_score(model, X_train_s, y_train, cv=cv, scoring="neg_mean_squared_error").mean()
    cv_mae = -cross_val_score(model, X_train_s, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()

    model.fit(X_train_s, y_train)
    ytr = model.predict(X_train_s)
    yte = model.predict(X_test_s)

    abs_err[name] = np.abs(y_test - yte)

    rows.append([name,
                 r2_score(y_train, ytr),
                 r2_score(y_test,  yte),
                 mean_squared_error(y_test, yte),
                 mean_absolute_error(y_test, yte),
                 cv_r2, cv_mse, cv_mae])

results = pd.DataFrame(rows, columns=["Model","Train R²","Test R²","MSE","MAE","CV_R²","CV_MSE","CV_MAE"])
print("\n=== Task 1: Model Performance Summary (Portfolio target) ===")
print(results)

# t-tests on |errors|
from itertools import combinations
pvals = []
for (m1, e1), (m2, e2) in combinations(abs_err.items(), 2):
    pvals.append({"pair": f"{m1} vs {m2}",
                  "p_value": ttest_rel(e1, e2, nan_policy='omit').pvalue})
pvals = pd.DataFrame(pvals)
print("\n=== Paired t-tests on |errors| (Test) ===")
print(pvals)

# Save + plot
results.to_csv(OUT / "task1_results_portfolio.csv", index=False, sep=";", decimal=",")
pvals.to_csv(OUT / "task1_ttests_portfolio.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(9,5))
x = np.arange(len(results["Model"]))
plt.bar(x-0.2, results["Train R²"], width=0.4, label="Train R²")
plt.bar(x+0.2, results["Test R²"],  width=0.4, label="Test R²")
plt.xticks(x, results["Model"]); plt.ylabel("R²"); plt.title("Train vs Test R² (Portfolio target)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "task1_r2_bars_portfolio.png", dpi=140); plt.close()

# ===================== Task 2.3 (Console + Files) — Random Forest SHAP =====================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Φάκελος εξόδων (αν δεν υπάρχει ήδη OUT)
try:
    OUT
except NameError:
    OUT = BASE_PATH / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)

# 1) Refit Random Forest στο TRAIN
rf_explain = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
rf_explain.fit(X_train_s, y_train)

feature_names = list(X_train.columns)

# 2) Built-in feature importance (plot + save)
imp_builtin = pd.Series(rf_explain.feature_importances_, index=feature_names).sort_values(ascending=False)
ax = imp_builtin.head(15)[::-1].plot(kind="barh", figsize=(8,7))
ax.set_title("Built-in Importance — Random Forest (Portfolio target)")
ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig(OUT / "task23_built_in_portfolio.png", dpi=150); plt.close()

# 3) SHAP (TreeExplainer) πάνω στο TEST
expl = shap.TreeExplainer(rf_explain)
sv   = expl.shap_values(np.asarray(X_test_s, dtype=np.float32))   # [n_test, n_features]
imp_shap = pd.Series(np.mean(np.abs(sv), axis=0), index=feature_names).sort_values(ascending=False)

# 3a) Plot SHAP top-15 (save)
ax = imp_shap.head(15)[::-1].plot(kind="barh", figsize=(8,7))
ax.set_title("SHAP Mean |Value| — Random Forest (Portfolio target)")
ax.set_xlabel("Mean |SHAP|")
plt.tight_layout(); plt.savefig(OUT / "task23_shap_portfolio.png", dpi=150); plt.close()

# 4) Σύγκριση built-in vs SHAP (CSV)
compare_df = pd.DataFrame({"built_in": imp_builtin, "shap": imp_shap})
compare_df.to_csv(OUT / "task23_compare_portfolio.csv")
print("✓ Saved plots & CSV in:", OUT)

# 5) Console summary (να φαίνεται “κάτω” στο Spyder)
TOPN = 10
shap_top = imp_shap.head(TOPN)
print("\n=== SHAP Top {} features (Random Forest, portfolio target) ===".format(TOPN))
for i, (feat, val) in enumerate(shap_top.items(), 1):
    print(f"{i:>2}. {feat:35s} | mean|SHAP| = {val:.6f}")

# 6) Προαιρετικά: σώσε και σε .txt
txt_path = OUT / "task23_shap_top_features.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("SHAP Top {} features (Random Forest, portfolio target)\n".format(TOPN))
    for i, (feat, val) in enumerate(shap_top.items(), 1):
        f.write(f"{i:>2}. {feat:35s} | mean|SHAP| = {val:.6f}\n")
print("✓ Saved SHAP top features →", txt_path)






# ------------------ 7) Weights (Equal & Mean-Variance 60d) ------------------
R = (returns[TICKERS] / 100.0).astype(float).dropna()

# Equal-weight (single-row + full time series for perf)
w_eq_vec = np.ones(len(TICKERS))/len(TICKERS)
pd.DataFrame([w_eq_vec], columns=TICKERS).to_csv(OUT / "weights_equal.csv", index=False)
w_eq_ts = pd.DataFrame(np.tile(w_eq_vec, (len(R), 1)), index=R.index, columns=TICKERS)

# Mean-Variance 60d (ridge shrinkage)
win = 60
W, D = [], []
for i in range(win, len(R)):
    window = R.iloc[i-win:i]
    mu = window.mean().values
    S  = window.cov().values + 1e-4*np.eye(len(TICKERS))
    try:
        x = np.linalg.solve(S, mu)
        w = x / x.sum() if x.sum()!=0 else np.ones_like(x)/len(x)
        # (optional long-only) w = np.clip(w, 0.0, 0.5); w = w / w.sum()
    except Exception:
        w = np.ones_like(mu)/len(mu)
    W.append(w); D.append(R.index[i])
w_mv = pd.DataFrame(W, index=D, columns=TICKERS)
w_mv.to_csv(OUT / "weights_mv60.csv")

def perf(Wdf, Rdf):
    # Ensure DataFrames & ευθυγράμμιση
    Wdf = pd.DataFrame(Wdf).copy()
    Rdf = pd.DataFrame(Rdf).copy()
    cols = [c for c in Rdf.columns if c in Wdf.columns]
    Wdf, Rdf = Wdf[cols], Rdf[cols]
    idx = Wdf.index.intersection(Rdf.index)
    Wdf, Rdf = Wdf.loc[idx], Rdf.loc[idx]

    # Κανονικοποίηση βαρών ανά ημέρα (αποφυγή μηδενικών/NaN)
    s = Wdf.sum(axis=1)
    s = s.replace(0, np.nan)
    Wdf = Wdf.div(s, axis=0).dropna(how="any")
    Rdf = Rdf.loc[Wdf.index]

    if len(Wdf) == 0:
        return pd.Series({"AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})

    # Daily portfolio return ως Series με index
    port = pd.Series(np.einsum('ij,ij->i', Wdf.values, Rdf.values), index=Wdf.index)

    # Stats
    T = port.size
    if T == 0:
        return pd.Series({"AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})

    ann = (1.0 + port).prod()**(252.0 / T) - 1.0
    vol = port.std() * np.sqrt(252.0)
    sharpe = ann / vol if vol and vol > 0 else np.nan

    eq = (1.0 + port).cumprod()
    mdd = (eq / eq.cummax() - 1.0).min()

    return pd.Series({"AnnReturn": ann, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": mdd})

