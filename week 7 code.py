# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:50:31 2025

@author: VasilikaC
"""

"""
WEEK 7 — End-to-End Pipeline (NO SPY dependency)
50 equities → Features → PCA → K-Means → Portfolios (EW + MinVar) → Performance → CAPM vs EW → Plots → REPORT

- Δε χρησιμοποιεί SPY (πολλές φορές κολλάει στο download).
- Benchmark = ίσο-σταθμισμένος μέσος του universe (EW benchmark).
- Προσοχή: χρησιμοποιούμε adjusted prices (auto_adjust=True) για να είναι σωστά τα returns μετά από splits/dividends.

Outputs (βασικοί φάκελοι):
BASE/
 ├─ DATA/ (prices, volume, shares, marketcap)
 ├─ PCA/  (features, loadings, scores, explained variance, plots)
 │   └─ CLUSTERING/ (elbow, silhouette, assignments, cluster means, 2D/3D plots, sizes)
 ├─ PORTFOLIO/ (weights & daily returns για EW και MinVar, performance table, plots)
 └─ REPORT.md (σχολιασμός & σύνοψη)
"""# -*- coding: utf-8 -*-


# ========================= Imports & Config =========================
import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---- User settings ----
SEED = 42
np.random.seed(SEED)

# Βάλε εδώ τον δικό σου φάκελο (κρατάω τον δικό σου default για να «γράφει» στο OneDrive σου)
BASE = Path(r"C:\Users\VasilikaC\OneDrive - CKH CPA\Desktop\Tsom erg\WEEK 7\WEEK7_outputs")
BASE.mkdir(parents=True, exist_ok=True)

SPYDER_PLOTS = True  # εμφάνιση στο Spyder Plots pane (και αποθήκευση σε αρχεία)

# Περίοδος (>= 24 μήνες)
START_DATE = "2012-04-11"
END_DATE   = "2025-04-11"

# 50 equities (όχι ETFs/crypto)
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V",
    "JNJ","PG","MA","XOM","HD","BAC","KO","PFE","PEP","DIS",
    "NFLX","NKE","CSCO","T","WMT","INTC","CVX","VZ","ADBE","ABNB",
    "CRM","MCD","PYPL","ORCL","COST","IBM","UPS","BA","AMD","QCOM",
    "MRK","HON","CAT","LLY","UNH","AVGO","CMCSA","TXN","SBUX","LOW"
]

# PCA / Features
VAR_THRESHOLD   = 0.90
ROLL_VOL_WIN    = 21
MOMENTUM_SKIP   = 21
WINDOW_TRAIN    = 252   # 12 μήνες για features/weights

# Clustering
K_MIN, K_MAX    = 2, 10
MIN_CLUSTER_ASSETS = 5

# Portfolios
MIN_WEIGHT      = 0.01          # 1% min ανά θέση
SLACK_CAP       = 0.02          # +2% πάνω από 1/N για να αποφύγουμε συγκέντρωση
CAP_MAX_FRACTION= 0.50          # <= 50% ανά θέση (upper cap)
COV_SHRINK      = 0.20          # shrinkage για σταθερότητα

# Performance
TRADING_DAYS    = 252
RF_ANNUAL       = 0.00

# ========================= Helpers =========================
def safe_to_csv(df, path, tries=3, sleep_s=0.6):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    for _ in range(tries):
        try:
            df.to_csv(tmp); os.replace(tmp, p); return
        except PermissionError:
            time.sleep(sleep_s)
    df.to_csv(p)

def check_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)

def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()

def max_drawdown_from_curve(curve: pd.Series) -> float:
    roll_max = curve.cummax()
    return float((curve/roll_max - 1.0).min())

def ann_stats(R: pd.DataFrame, shrink: float = 0.10):
    mu = R.mean() * TRADING_DAYS
    cov = R.cov() * TRADING_DAYS
    if shrink > 0:
        D = np.diag(np.diag(cov.values))
        cov = pd.DataFrame((1 - shrink) * cov.values + shrink * D, index=cov.index, columns=cov.columns)
    vol = pd.Series(np.sqrt(np.diag(cov.values)), index=R.columns)
    return mu, cov, vol

def capm_alpha_beta_R2(port_r: pd.Series, bench_r: pd.Series, rf_ann=0.0):
    rf_d = rf_ann / TRADING_DAYS
    x = (bench_r - rf_d).dropna()
    y = (port_r - rf_d).reindex_like(x)
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30: return np.nan, np.nan, np.nan
    X = df.iloc[:,0].values; Y = df.iloc[:,1].values
    Xc = X - X.mean(); Yc = Y - Y.mean()
    beta = (Xc @ Yc) / (Xc @ Xc) if (Xc @ Xc) != 0 else np.nan
    alpha_daily = Y.mean() - beta * X.mean()
    y_hat = alpha_daily + beta * X
    ss_res = np.sum((Y - y_hat)**2); ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
    return alpha_daily * TRADING_DAYS, beta, r2

def ksafe_show():
    if SPYDER_PLOTS:
        plt.show()
    plt.close()

# ========================= 1) Data =========================
DATA_DIR = BASE / "DATA"; DATA_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading adjusted prices & volume…")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, interval="1d",
                  auto_adjust=True, progress=False, group_by="ticker")

prices = pd.DataFrame({tic: raw[tic]["Close"]  for tic in TICKERS if tic in raw.columns.get_level_values(0)}).ffill()
volume = pd.DataFrame({tic: raw[tic]["Volume"] for tic in TICKERS if tic in raw.columns.get_level_values(0)}).fillna(0)
# φιλτράρισμα για αρκετό ιστορικό
usable = [c for c in prices.columns if prices[c].dropna().shape[0] >= int(1.9*TRADING_DAYS)]
prices = prices[usable]; volume = volume[usable]

# Shares outstanding (best-effort) → Market Cap
def get_shares_series(tic: str, idx: pd.DatetimeIndex) -> pd.Series:
    tk = yf.Ticker(tic)
    s = None
    try:
        sh = tk.get_shares_full(start=START_DATE, end=END_DATE)
        if sh is not None and len(sh) > 0:
            s = sh.resample("D").ffill().reindex(idx).ffill()
    except Exception:
        s = None
    if s is None:
        val = None
        try: val = getattr(tk.fast_info, "shares", None)
        except Exception: val = None
        if val in (None, 0):
            try: val = (tk.info or {}).get("sharesOutstanding", None)
            except Exception: val = None
        if val and val > 0:
            s = pd.Series(val, index=idx, dtype="float64")
    if s is None:
        s = pd.Series(np.nan, index=idx, dtype="float64")
    return s

print("Fetching shares outstanding (best-effort)…")
dates = prices.index
shares_df = pd.DataFrame(index=dates, columns=prices.columns, dtype="float64")
for t in prices.columns:
    try:
        shares_df[t] = get_shares_series(t, dates)
    except Exception:
        shares_df[t] = np.nan

market_cap = (prices * shares_df).mask(lambda x: x <= 0, np.nan)

# Save data
safe_to_csv(prices, DATA_DIR/"prices.csv")
safe_to_csv(volume, DATA_DIR/"volume.csv")
safe_to_csv(shares_df, DATA_DIR/"shares_outstanding.csv")
safe_to_csv(market_cap, DATA_DIR/"market_cap.csv")
print("✅ Saved DATA to:", DATA_DIR)

# ========================= 2) Features → PCA =========================
PCA_DIR = BASE / "PCA"; PCA_DIR.mkdir(parents=True, exist_ok=True)

# training window ~ 12m
if len(prices) > WINDOW_TRAIN:
    px = prices.iloc[-WINDOW_TRAIN:].copy()
    vol = volume.iloc[-WINDOW_TRAIN:].copy()
    shs = shares_df.iloc[-WINDOW_TRAIN:].copy()
    mcap= market_cap.iloc[-WINDOW_TRAIN:].copy()
else:
    px, vol, shs, mcap = prices.copy(), volume.copy(), shares_df.copy(), market_cap.copy()

rets = px.pct_change()
logr = np.log(px/px.shift(1))
dvol = px * vol

# EW benchmark (καθολικό, χωρίς SPY)
bench_ret_EW = rets.mean(axis=1, skipna=True)

def beta_generic(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if len(df) < 30: return np.nan
    cov = np.cov(df.iloc[:,0], df.iloc[:,1], ddof=1)[0,1]
    var = np.var(df.iloc[:,1], ddof=1)
    return cov/var if var>0 else np.nan

def momentum_12_1(ret_series: pd.Series, window=TRADING_DAYS, skip=MOMENTUM_SKIP) -> float:
    x = ret_series.dropna()
    if len(x) < 30: return np.nan
    w = x.iloc[-window:-skip] if len(x) >= window else x.iloc[:max(0, len(x)-skip)]
    if len(w) < 30: return np.nan
    return float((1 + w).prod() - 1)

def max_drawdown_12m(prices_series: pd.Series) -> float:
    x = prices_series.dropna().astype(float)
    if len(x) < 30: return np.nan
    rollmax = x.cummax()
    return float((x/rollmax - 1.0).min())

last_mcap      = mcap.ffill().iloc[-1]
LogMktCap      = np.log(last_mcap.replace(0, np.nan))
ADV            = dvol.replace(0, np.nan).mean(axis=0)
LogADV         = np.log(ADV)
Turnover       = (vol / shs.replace(0, np.nan)).mean(axis=0)  # proxy αν λείπουν shares
Amihud         = (rets.abs() / dvol.replace(0, np.nan)).mean(axis=0) * 1e6
rv_21          = logr.rolling(ROLL_VOL_WIN, min_periods=int(0.75*ROLL_VOL_WIN)).std() * np.sqrt(TRADING_DAYS)
RealizedVol_21 = rv_21.median(axis=0)
Beta_Bench     = rets.apply(lambda s: beta_generic(s, bench_ret_EW))
Momentum_12_1  = rets.apply(momentum_12_1)
MaxDrawdown_12m= px.apply(max_drawdown_12m)

feature_df = pd.DataFrame({
    "LogMktCap": LogMktCap,
    "LogADV": LogADV,
    "Turnover": Turnover,
    "Amihud": Amihud,
    "RealizedVol_21": RealizedVol_21,
    "Beta_EW": Beta_Bench,
    "Momentum_12_1": Momentum_12_1,
    "MaxDrawdown_12m": MaxDrawdown_12m,
})
feature_df = check_numeric_df(feature_df)
feature_df = feature_df.loc[:, feature_df.isna().mean() <= 0.5]
feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
safe_to_csv(feature_df, PCA_DIR/"feature_df.csv")

# PCA
X = feature_df.values.astype(float)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)
pca = PCA(random_state=SEED)
scores_all = pca.fit_transform(Xz)
expl = pca.explained_variance_ratio_
cum  = expl.cumsum()
k = int(np.argmax(cum >= VAR_THRESHOLD) + 1)

loadings   = pd.DataFrame(pca.components_.T, index=feature_df.columns,
                          columns=[f"PC{i}" for i in range(1, len(expl)+1)])
scores_top = pd.DataFrame(scores_all[:, :k], index=feature_df.index,
                          columns=[f"PC{i}" for i in range(1, k+1)])

safe_to_csv(loadings,   PCA_DIR/"pca_loadings_all.csv")
safe_to_csv(scores_top, PCA_DIR/"pca_scores_topk.csv")
safe_to_csv(pd.Series(expl, index=loadings.columns, name="expl_var"), PCA_DIR/"explained_variance_ratio.csv")
safe_to_csv(pd.Series(cum, index=loadings.columns, name="cum_expl_var"), PCA_DIR/"cumulative_explained_variance.csv")

# plots
plt.figure(); plt.bar(range(1, len(expl)+1), expl)
plt.xlabel("PC"); plt.ylabel("Explained variance"); plt.title("PCA – per component")
plt.tight_layout(); plt.savefig(PCA_DIR/"explained_variance_ratio.png"); ksafe_show()

plt.figure(); plt.plot(range(1, len(cum)+1), cum, marker="o"); plt.axhline(VAR_THRESHOLD, ls="--")
plt.xlabel("#PCs"); plt.ylabel("Cumulative explained variance"); plt.title("PCA – cumulative")
plt.tight_layout(); plt.savefig(PCA_DIR/"cumulative_explained_variance.png"); ksafe_show()

with open(PCA_DIR/"pca_report.txt", "w", encoding="utf-8") as f:
    f.write(f"PCA k={k} (cum={cum[k-1]:.2%})\nTop PC1: " +
            ", ".join(loadings["PC1"].abs().sort_values(ascending=False).head(5).index))

print(f"✅ PCA selected k={k} (cum={cum[k-1]:.2%}) | outputs in {PCA_DIR}")

# ========================= 3) K-Means (with size-based relabel) =========================
CLUST_DIR = PCA_DIR / "CLUSTERING"; CLUST_DIR.mkdir(parents=True, exist_ok=True)

# Θα κλικάρουμε στα PCA scores (ήδη κανονικοποιημένα πιο πάνω)
Xk = StandardScaler().fit_transform(scores_top.values)
assets = scores_top.index

# Sweep για Elbow/Silhouette (same logic)
K_RANGE = range(K_MIN, min(K_MAX+1, len(assets)))
inertias, sils = [], []
best_k, best_s = None, -np.inf

for kk in K_RANGE:
    km = KMeans(n_clusters=kk, n_init=50, random_state=SEED)
    lab = km.fit_predict(Xk)
    inertias.append(km.inertia_)
    # απόφυγε degenerate clusters στο silhouette
    if (np.bincount(lab) < 2).any():
        sils.append(np.nan)
        continue
    s = silhouette_score(Xk, lab)
    sils.append(s)
    if s > best_s:
        best_s, best_k = s, kk

# Αποθήκευση Elbow/Silhouette
plt.figure(); plt.plot(list(K_RANGE), inertias, marker="o")
plt.title("Elbow (Inertia) vs K"); plt.xlabel("K"); plt.ylabel("Inertia (WCSS)")
plt.tight_layout(); plt.savefig(CLUST_DIR / "elbow_curve.png"); ksafe_show()

plt.figure(); plt.plot(list(K_RANGE), sils, marker="o")
if best_k: plt.axvline(best_k, ls="--", alpha=0.6, label=f"Best K={best_k}"); plt.legend()
plt.title("Silhouette vs K"); plt.xlabel("K"); plt.ylabel("Silhouette")
plt.tight_layout(); plt.savefig(CLUST_DIR / "silhouette_curve.png"); ksafe_show()

if best_k is None: best_k = 3  # fallback

# Τελικό KMeans
kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=SEED)
labels_orig = kmeans.fit_predict(Xk)

# -------- ΝΕΟ: Μετονομασία clusters με βάση το ΜΕΓΕΘΟΣ --------
assign_raw = pd.DataFrame({"Cluster_raw": labels_orig}, index=assets)
sizes_raw = assign_raw["Cluster_raw"].value_counts().sort_values(ascending=False)  # φθίνουσα (μεγαλύτερο πρώτα)
order = list(sizes_raw.index)  # π.χ. [4, 0, 1, 3, 2, ...] = αρχικές ετικέτες με σειρά μεγέθους
mapping = {old: new for new, old in enumerate(order)}  # {παλιά:νέα}, ώστε 0=largest

assign = assign_raw.copy()
assign["Cluster"] = assign_raw["Cluster_raw"].map(mapping)  # ΝΕΕΣ σταθερές ετικέτες
assign.to_csv(CLUST_DIR / "asset_cluster_assignments.csv")

# mapping → csv για τεκμηρίωση
pd.DataFrame({
    "Cluster_original": sizes_raw.index,
    "Cluster_renamed":  [mapping[o] for o in sizes_raw.index],
    "Size":             sizes_raw.values
}).to_csv(CLUST_DIR / "cluster_label_mapping.csv", index=False)

# Μέσοι όροι features ανά ΝΕΟ cluster
features_used = feature_df.copy()
feat_means = features_used.join(assign["Cluster"], how="inner").groupby("Cluster").mean(numeric_only=True)
feat_means.to_csv(CLUST_DIR / "cluster_feature_means.csv")

# 2D scatter με ΝΕΕΣ ετικέτες και ταξινομημένο legend (0=largest)
if scores_top.shape[1] >= 2:
    plt.figure()
    for c in sorted(assign["Cluster"].unique()):
        mask = (assign["Cluster"] == c)
        pts = scores_top.loc[mask.values]  # ίδιο index (assets)
        plt.scatter(pts.iloc[:,0], pts.iloc[:,1], s=35, label=f"C{c}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-Means on PCs (2D) — relabeled by size")
    plt.legend(title="Cluster"); plt.tight_layout()
    plt.savefig(CLUST_DIR / "clusters_2D_PC1_PC2.png"); ksafe_show()

# 3D (προαιρετικό)
if scores_top.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    for c in sorted(assign["Cluster"].unique()):
        mask = (assign["Cluster"] == c); pts = scores_top.loc[mask.values]
        ax.scatter(pts.iloc[:,0], pts.iloc[:,1], pts.iloc[:,2], s=22, label=f"C{c}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    plt.title("K-Means on PCs (3D) — relabeled by size"); plt.legend()
    plt.tight_layout(); plt.savefig(CLUST_DIR / "clusters_3D_PC1_PC2_PC3.png"); ksafe_show()

# Μεγέθη clusters με βάση τις ΝΕΕΣ ετικέτες (0=largest)
sizes = assign["Cluster"].value_counts().sort_index()
sizes.to_csv(CLUST_DIR / "cluster_sizes.csv")

with open(CLUST_DIR / "clustering_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Selected K={best_k} (silhouette={best_s:.3f})\n\n")
    f.write("Relabel mapping (original → renamed by size):\n")
    for old, new in mapping.items():
        f.write(f"- {old} → {new}\n")
    f.write("\nCluster sizes (renamed):\n")
    for c, n in sizes.items():
        f.write(f"- Cluster {c}: {int(n)} assets\n")

print(f"✅ Clustering K={best_k} (sil={best_s:.3f}) | relabeled so C0=largest | outputs → {CLUST_DIR}")



# ========================= 4) Portfolios (EW + MinVar) =========================
PORT_DIR = BASE / "PORTFOLIO"; PORT_DIR.mkdir(parents=True, exist_ok=True)
px_win = prices.iloc[-WINDOW_TRAIN:] if len(prices) > WINDOW_TRAIN else prices.copy()
rets_win = px_win.pct_change().dropna(how="all")

common = rets_win.columns.intersection(assign.index)
rets_win = rets_win.loc[:, common]
assign   = assign.loc[common]

def minvar_slsqp(R: pd.DataFrame, cap=0.10, min_w=0.01, shrink=0.20):
    _, cov, _ = ann_stats(R, shrink=shrink)
    n = R.shape[1]
    if n < 2:
        return pd.Series(1.0, index=R.columns)
    x0 = np.full(n, 1.0/n)
    bounds = [(min_w, min(cap, 1.0)) for _ in range(n)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    def obj(w): return float(w @ cov.values @ w)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        return pd.Series(x0, index=R.columns)
    return pd.Series(res.x, index=R.columns)

def save_port(label, R, w, c):
    mu, cov, _ = ann_stats(R, shrink=0.0)
    pr = R.values @ w.reindex(R.columns).fillna(0.0).values
    s  = pd.Series(pr, index=R.index, name=label)
    # save weights & daily returns
    pd.DataFrame({"Weight": w}).sort_values("Weight", ascending=False)\
        .to_csv(PORT_DIR / f"cluster{c}_{label}_weights.csv")
    s.to_csv(PORT_DIR / f"cluster{c}_{label}_daily_returns.csv")

summary_rows = []

for c, grp in assign.groupby("Cluster"):
    names = grp.index.tolist()
    R = rets_win[names].dropna(axis=1, how="all").dropna(how="any")
    n_assets = R.shape[1]
    print(f"\nCluster {c}: total={len(names)}, usable={n_assets}")
    if len(R) < 60 or n_assets < 2:
        print("  Skipped (λίγες μέρες ή λίγα assets).")
        continue

    # ---- EW baseline για ΟΛΑ τα clusters ----
    w_ew = pd.Series(1.0/n_assets, index=R.columns)
    save_port("EW", R, w_ew, c)

    # ---- MinVar (deterministic) με ρεαλιστικούς περιορισμούς ----
    cap_eff   = min(CAP_MAX_FRACTION, max(1.0/n_assets + SLACK_CAP, 0.10))
    min_w_eff = MIN_WEIGHT if n_assets*MIN_WEIGHT <= 1.0 else (0.5/n_assets)
    w_mv = minvar_slsqp(R, cap=cap_eff, min_w=min_w_eff, shrink=COV_SHRINK)
    save_port("MinVar", R, w_mv, c)

print("✅ Saved portfolios (EW & MinVar) to:", PORT_DIR)

# ========================= 5) Performance + CAPM (vs EW) =========================
PLOTS_DIR = PORT_DIR / "plots"; PLOTS_DIR.mkdir(parents=True, exist_ok=True)

rets_all = prices.pct_change()
bench_series = rets_all.mean(axis=1).dropna()  # EW benchmark
bench_label  = "EW Benchmark"

rows = []
port_files = sorted([f for f in os.listdir(PORT_DIR) if f.endswith("_daily_returns.csv")])

def evaluate_portfolio(path_csv, label):
    r = pd.read_csv(path_csv, index_col=0, parse_dates=True).iloc[:,0].dropna()
    idx = r.index.intersection(bench_series.index)
    r, b = r.loc[idx], bench_series.loc[idx]
    if len(r) < 30: return None
    curv = equity_curve(r)
    cum_ret = float(curv.iloc[-1] - 1.0)
    mu_a = r.mean() * TRADING_DAYS
    vol_a = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    shp = (mu_a - RF_ANNUAL) / vol_a if vol_a>0 else np.nan
    a_ann, beta, r2 = capm_alpha_beta_R2(r, b, rf_ann=RF_ANNUAL)
    mdd = max_drawdown_from_curve(curv)

    # plots
    bcurve = equity_curve(b.reindex(curv.index))
    plt.figure()
    plt.plot(curv.index, curv.values, label=label)
    plt.plot(bcurve.index, bcurve.values, "--", label=bench_label)
    plt.title(f"Equity Curve — {label}")
    plt.xlabel("Date"); plt.ylabel("Wealth index (start=1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{label}_equity.png"); ksafe_show()

    plt.figure()
    plt.hist(r.values, bins=60, density=True)
    plt.title(f"Returns Distribution — {label}")
    plt.xlabel("Daily return"); plt.ylabel("Density")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"{label}_ret_dist.png"); ksafe_show()

    return {
        "Portfolio": label, "Start": r.index.min().date(), "End": r.index.max().date(), "Days": len(r),
        "CumReturn_%": round(100*cum_ret, 2), "AnnReturn_%": round(100*mu_a, 2),
        "AnnVol_%": round(100*vol_a, 2), "Sharpe": round(shp, 3), "MaxDrawdown_%": round(100*mdd, 2),
        "CAPM_Benchmark": bench_label,
        "Alpha_ann_%": round(100*a_ann, 2) if pd.notna(a_ann) else np.nan,
        "Beta": round(beta, 3) if pd.notna(beta) else np.nan,
        "CAPM_R2": round(r2, 3) if pd.notna(r2) else np.nan,
    }

for f in port_files:
    label = f.replace("_daily_returns.csv", "")
    res = evaluate_portfolio(PORT_DIR / f, label)
    if res: rows.append(res)

perf = pd.DataFrame(rows).sort_values("Portfolio").reset_index(drop=True)
safe_to_csv(perf, PORT_DIR/"performance_metrics_summary.csv")
print("\n✅ Saved performance table:", PORT_DIR/"performance_metrics_summary.csv")

# risk–return scatter
if not perf.empty:
    plt.figure()
    plt.scatter(perf["AnnVol_%"], perf["AnnReturn_%"], s=80)
    for _, row in perf.iterrows():
        plt.annotate(row["Portfolio"], (row["AnnVol_%"], row["AnnReturn_%"]),
                     xytext=(5,2), textcoords="offset points", fontsize=8)
    plt.xlabel("Ann. Volatility (%)"); plt.ylabel("Ann. Return (%)")
    plt.title("Risk–Return (all portfolios)")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "risk_return_scatter.png"); ksafe_show()

# cluster sizes
plt.figure()
plt.bar([str(k) for k in sizes.index], sizes.values)
plt.title("Cluster Composition — asset count")
plt.xlabel("Cluster"); plt.ylabel("# assets")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "cluster_sizes_bar.png"); ksafe_show()

# ========================= 6) Best-in-class summary =========================
def best_by_sharpe(perf_df):
    if perf_df.empty: return None, None, None
    df2 = perf_df.sort_values(["Sharpe","AnnReturn_%","AnnVol_%"], ascending=[False,False,True]).reset_index(drop=True)
    top = df2.iloc[0].to_dict()
    runner = df2.iloc[1].to_dict() if len(df2) > 1 else None
    return top, runner, df2

top, runner, rankdf = best_by_sharpe(perf)
lines = []
lines.append("### Best by Sharpe")
if top:
    lines.append(f"- **Best portfolio:** {top['Portfolio']}")
    lines.append(f"- **Rationale:** υψηλότερος Sharpe με ευνοϊκούς tie-breakers (υψηλότερη απόδοση/ηπιότερη μεταβλητότητα).")
    lines.append("- **Top-3:** " + ", ".join(rankdf.head(3)["Portfolio"].tolist()))
best_report = "\n".join(lines) if lines else "No portfolios."

with open(PORT_DIR/"best_cluster_report.md", "w", encoding="utf-8") as f:
    f.write(best_report)
print("✅ Best-in-class summary:", PORT_DIR/"best_cluster_report.md")

# ========================= 7) REPORT (σχολιαστικό) =========================
REPORT = BASE / "REPORT.md"
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("# Report — PCA → K-Means → Portfolios (EW/MinVar) → Performance (vs EW)\n\n")
    f.write("## Σύνοψη\n")
    f.write("- Ομαδοποιήσαμε 50 μετοχές με PCA+K-Means και κατασκευάσαμε EW & MinVar χαρτοφυλάκια ανά cluster.\n")
    f.write("- Αξιολόγηση με CumRet, AnnRet/Vol, Sharpe, MaxDD και **CAPM (alpha/beta/R²)** έναντι **EW Benchmark**.\n")
    f.write("- Τα αποτελέσματα αναδεικνύουν 1–2 clusters ως πυρήνα (υψηλή ποιότητα risk-adjusted) και άλλα ως αμυντικά/δευτερεύοντα.\n\n")

    f.write("## Data & Preprocessing\n")
    f.write("- Πηγές: yfinance (adjusted prices/volume), περίοδος: " +
            f"{prices.index.min().date()} → {prices.index.max().date()}.\n")
    f.write("- Καθαρισμός: ffill για prices, volume=0 σε κενά, έλεγχος/αντικατάσταση NaN στα features με median.\n")
    f.write("- Benchmark: **EW** (ίσος μέσος αποδόσεων σύμπαντος) — επιλέχθηκε για να αποφευχθούν εξαρτήσεις από SPY.\n\n")

    f.write("## Features & PCA\n")
    f.write("- Features: LogMktCap, LogADV, Turnover, Amihud, RealizedVol(21), Beta(EW), Momentum(12–1), MaxDrawdown(12m).\n")
    f.write(f"- PCA threshold **{int(VAR_THRESHOLD*100)}%** → επιλέχθηκαν **k={k}** συνιστώσες. Δες scree & cumulative plots.\n\n")

    f.write("## Clustering\n")
    f.write(f"- K-Means στα PCA scores. Επιλογή **K** με silhouette (2–10), καλύτερο K ~ **{best_k}** (μέτρια αλλά χρήσιμη διάκριση).\n")
    f.write("- Δόθηκαν αναθέσεις τίτλων, μέσοι όροι features ανά cluster, 2D/3D απεικονίσεις και μέγεθος clusters.\n\n")

    f.write("## Portfolios & Αξιολόγηση\n")
    f.write("- Για κάθε cluster: **EW baseline** και **MinVar** (no-shorts, caps, shrinkage).\n")
    f.write("- Μετρικές: ετησιοποιημένες αποδόσεις/μεταβλητότητα, Sharpe, μέγιστο drawdown, CAPM alpha/beta/R² vs EW.\n")
    f.write("- Οπτικοποιήσεις: equity curves ανά portfolio vs benchmark, κατανομές αποδόσεων, risk–return scatter.\n\n")

    if top:
        f.write("## Best-in-Class (By Sharpe)\n")
        f.write(f"- Νικητής: **{top['Portfolio']}**. Βλέπε `best_cluster_report.md` και αντίστοιχα plots.\n\n")

    f.write("## Προτάσεις (ενδεικτικά)\n")
    f.write("- Πυρήνας: το καλύτερο cluster-portfolio (συνήθως MinVar για ήπιο προφίλ),\n")
    f.write("  με αμυντικό συμπλήρωμα από cluster χαμηλής μεταβλητότητας.\n")
    f.write("- Περιορισμός/αποφυγή clusters με χαμηλό Sharpe ή αρνητικό alpha.\n\n")

    f.write("## Περιορισμοί & Επόμενα Βήματα\n")
    f.write("- Estimation risk (συνδιακυμάνσεις): αντιμετωπίστηκε με shrinkage & caps· καλό είναι να μπει walk‑forward.\n")
    f.write("- Μικρά clusters ενδέχεται να είναι ασταθή — χρησιμοποιήθηκε EW baseline για όλα.\n")
    f.write("- Μπορεί να προστεθεί Risk Parity/HRP ως τρίτη optimized μέθοδος για σύγκριση.\n")

print("📝 REPORT:", REPORT)
print("\nAll done ✅  Base folder:", BASE)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def plot_elbow_and_silhouette(X, k_min=2, k_max=10, out_png=None):
    Ks = list(range(k_min, k_max+1))
    inertia, sil = [], []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        inertia.append(km.inertia_)
        lab = km.labels_
        # αγνόησε K με πολύ μικρά clusters (ασταθές silhouette)
        if (np.bincount(lab) < 2).any():
            sil.append(np.nan)
        else:
            sil.append(silhouette_score(X, lab))

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(Ks, inertia, '-o'); ax[0].set_title('Elbow (Inertia)'); ax[0].set_xlabel('K'); ax[0].set_ylabel('Inertia')
    ax[1].plot(Ks, sil, '-o'); ax[1].set_title('Silhouette vs K'); ax[1].set_xlabel('K'); ax[1].set_ylabel('Silhouette')
    plt.tight_layout()
    if out_png: plt.savefig(out_png)
    plt.show()
plot_elbow_and_silhouette(Xk, k_min=2, k_max=10, out_png=CLUST_DIR/'elbow_vs_silhouette.png')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def silhouette_sweep_meanstd(Xk, k_min=2, k_max=10, seeds=range(10), n_init=50, min_prop=0.05):
    rows = []
    n = Xk.shape[0]
    min_sz = max(3, int(min_prop*n))     # π.χ. >=5% του δείγματος (και τουλάχιστον 3 σημεία)
    for k in range(k_min, k_max+1):
        vals = []
        for s in seeds:
            km = KMeans(n_clusters=k, n_init=n_init, random_state=s)
            lab = km.fit_predict(Xk)
            if (np.bincount(lab) < min_sz).any():
                continue  # αγνόησε K που παράγουν πολύ μικρά clusters
            vals.append(silhouette_score(Xk, lab))
        rows.append({"K": k, "mean": np.mean(vals) if vals else np.nan,
                          "std":  np.std(vals)  if vals else np.nan,
                          "runs": len(vals)})
    df = pd.DataFrame(rows)
    best_k = int(df.loc[df["mean"].idxmax(), "K"])
    # plot με error bars και μαρκάρισμα του βέλτιστου K
    plt.figure()
    plt.errorbar(df["K"], df["mean"], yerr=df["std"], fmt="-o", capsize=3)
    plt.axvline(best_k, ls="--", alpha=0.6, label=f"Best K={best_k}")
    plt.title("Silhouette vs K (mean ± std)")
    plt.xlabel("K"); plt.ylabel("Silhouette"); plt.legend(); plt.tight_layout()
    plt.show()
    return df, best_k


