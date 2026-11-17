# %% [markdown]
# # Choosing Alarm Thresholds for Machine Failure Prediction (ES 335)
# Full pipeline: split -> preprocessing -> models -> threshold/cost analysis -> figures -> JSON for the website.
# Run top to bottom (`python src/pipeline.py` from the repo root, or run `notebooks/analysis.ipynb`).

# %%
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, f1_score, accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

SEED = 42
try:    ROOT = Path(__file__).resolve().parent.parent
except NameError: ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, DATA, FIG = ROOT / "data/raw/ai4i2020.csv", ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True); (ROOT / "site_data").mkdir(exist_ok=True)

TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RATIOS = np.unique(np.concatenate([np.round(np.geomspace(1, 200, 41), 1), [1, 2, 5, 10, 20, 50, 100, 200]]))      # cost of a missed failure / cost of a false alarm
R_REF = 10.0                                        # reference ratio used for error analysis
TGRID = np.round(np.linspace(0, 1, 1001), 3)        # candidate alarm thresholds
FPR_BUDGETS = [0.01, 0.02, 0.05, 0.10]

# palette shared with the website
INK, ALARM, CALM, GREY = "#16233B", "#D9480F", "#0F7B8A", "#8A94A6"
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
              "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold", "figure.dpi": 130})

# %% [markdown]
# ## 1. Load and split (the split is made first; the test set is touched once at the end)

# %%
raw = pd.read_csv(RAW, encoding="utf-8-sig")
raw.columns = raw.columns.str.replace(r"\s*\[.*\]", "", regex=True)
y_all = raw["Machine failure"]
train, test = train_test_split(raw, test_size=0.2, stratify=y_all, random_state=SEED)
train, test = train.sort_values("UDI"), test.sort_values("UDI")
train.to_csv(DATA / "train.csv", index=False); test.to_csv(DATA / "test.csv", index=False)
print("train", train.shape, train["Machine failure"].sum(), "| test", test.shape, test["Machine failure"].sum())

# %% [markdown]
# ## 2. Features. Failure-type columns and IDs are never used as inputs.
# The three physics features are plain row-wise formulas (no parameters learned from data), so they cannot leak.

# %%
def add_phys(df):
    d = df.copy()
    d["Power (W)"] = d["Torque"] * d["Rotational speed"] * 2 * np.pi / 60
    d["Temp diff (K)"] = d["Process temperature"] - d["Air temperature"]
    d["Wear x torque"] = d["Tool wear"] * d["Torque"]
    return d

NUM_RAW = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
NUM_ENG = NUM_RAW + ["Power (W)", "Temp diff (K)", "Wear x torque"]
FEATS = {"raw": NUM_RAW, "eng": NUM_ENG}
tr, te = add_phys(train), add_phys(test)
ytr, yte = tr["Machine failure"].values, te["Machine failure"].values

def make_pipe(clf, feat):
    pre = ColumnTransformer([("num", StandardScaler(), FEATS[feat]), ("cat", OneHotEncoder(), ["Type"])])
    return Pipeline([("pre", pre), ("clf", clf)])   # scaling/encoding are fitted inside each CV training fold

MODELS = {
    "logreg": ("Logistic regression", LogisticRegression(class_weight="balanced", max_iter=3000),
               {"clf__C": [0.1, 1, 10]}),
    "tree":   ("Decision tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
               {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 20]}),
    "forest": ("Random forest", RandomForestClassifier(n_estimators=80, class_weight="balanced_subsample",
               random_state=SEED, n_jobs=1),
               {"clf__max_depth": [8], "clf__min_samples_leaf": [5, 15]}),
    "gboost": ("Gradient boosting", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
               {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [8, 15], "clf__min_samples_leaf": [20, 50]}),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# %% [markdown]
# ## 3. Threshold helpers (thresholds are always chosen on out-of-fold training predictions, never on the test set)

# %%
def counts_at(y, p, t):
    pred = p >= t
    return dict(tp=int((pred & (y == 1)).sum()), fp=int((pred & (y == 0)).sum()),
                fn=int((~pred & (y == 1)).sum()), tn=int((~pred & (y == 0)).sum()))

def fn_fp_grid(y, p):
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, TGRID, side="left")              # positives with p < t
    fp = len(neg) - np.searchsorted(neg, TGRID, side="left")   # negatives with p >= t
    return fn, fp

def optimal_thresholds(y, p):
    fn, fp = fn_fp_grid(y, p); out = []
    for r in RATIOS:
        cost = r * fn + fp; idx = np.flatnonzero(cost == cost.min())
        out.append(float(TGRID[idx[len(idx) // 2]]))           # middle of the tied range
    return out

def threshold_for_fpr(y, p, budget):
    _, fp = fn_fp_grid(y, p); ok = np.flatnonzero(fp / (y == 0).sum() <= budget)
    return float(TGRID[ok[0]])

def ap(y, p): return float(average_precision_score(y, p))

# %% [markdown]
# ## 4. Baselines + candidate models: tune with 5-fold stratified CV (metric: PR-AUC) on the training set only

# %%
res, probs_test, oof = {}, {}, {}
# baseline 1: majority class ("always no failure")
dummy = Pipeline([("pre", ColumnTransformer([("num", "passthrough", NUM_RAW)])), ("clf", DummyClassifier(strategy="prior"))]).fit(tr, ytr)
pd_ = dummy.predict_proba(te)[:, 1]
baseline = dict(accuracy=float(accuracy_score(yte, np.zeros_like(yte))), test_ap=ap(yte, pd_),
                prevalence=float(yte.mean()), recall=0.0)

for key, (label, clf, grid) in MODELS.items():
    for feat in FEATS:
        name = f"{key}__{feat}"; print("fitting", name, flush=True)
        gs = GridSearchCV(make_pipe(clf, feat), grid, scoring="average_precision", cv=skf, n_jobs=1, refit=True).fit(tr, ytr)
        bi = gs.best_index_
        folds = [float(gs.cv_results_[f"split{i}_test_score"][bi]) for i in range(5)]
        best = gs.best_estimator_
        # raw out-of-fold predictions -> used for threshold choice
        p_oof = cross_val_predict(clone(best), tr, ytr, cv=skf, method="predict_proba")[:, 1]
        p_te = best.predict_proba(te)[:, 1]
        # calibration: choose sigmoid vs isotonic by out-of-fold Brier on the training set
        cal = {}
        for m in ["sigmoid", "isotonic"]:
            c = CalibratedClassifierCV(clone(best), method=m, cv=5)
            po = cross_val_predict(c, tr, ytr, cv=skf, method="predict_proba")[:, 1]
            cal[m] = (brier_score_loss(ytr, po), po)
        m_best = min(cal, key=lambda k: cal[k][0]); p_oof_c = cal[m_best][1]
        p_te_c = CalibratedClassifierCV(clone(best), method=m_best, cv=5).fit(tr, ytr).predict_proba(te)[:, 1]
        probs_test[name] = (p_te, p_te_c); oof[name] = (p_oof, p_oof_c)
        res[name] = dict(model=key, feat=feat, label=label, params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                         cv_ap_mean=float(np.mean(folds)), cv_ap_std=float(np.std(folds)), cv_folds=folds,
                         cal_method=m_best, oof_brier_raw=float(brier_score_loss(ytr, p_oof)), oof_brier_cal=float(cal[m_best][0]),
                         thr_raw=optimal_thresholds(ytr, p_oof), thr_cal=optimal_thresholds(ytr, p_oof_c))

# %% [markdown]
# ## 5. Pick the winner by CV PR-AUC only; then look at the test set once

# %%
winner = max(res, key=lambda k: res[k]["cv_ap_mean"]); print("WINNER (by CV PR-AUC):", winner)
for name, r in res.items():
    p, pc = probs_test[name]; po, poc = oof[name]
    r["test"] = dict(ap=ap(yte, p), roc=float(roc_auc_score(yte, p)), brier_raw=float(brier_score_loss(yte, p)),
                     brier_cal=float(brier_score_loss(yte, pc)), ap_cal=ap(yte, pc),
                     acc_at_half=float(accuracy_score(yte, p >= .5)), f1_at_half=float(f1_score(yte, p >= .5)),
                     recall_at_half=float(((p >= .5) & (yte == 1)).sum() / yte.sum()),
                     precision_at_half=float(((p >= .5) & (yte == 1)).sum() / max((p >= .5).sum(), 1)))
    r["fpr_budget"] = []
    for b in FPR_BUDGETS:
        t = threshold_for_fpr(ytr, po, b); c = counts_at(yte, p, t)
        r["fpr_budget"].append(dict(budget=b, thr=t, recall=c["tp"] / yte.sum(), test_fpr=c["fp"] / (yte == 0).sum(), **c))
    rr = RATIOS.tolist().index(R_REF); t = r["thr_raw"][rr]; c = counts_at(yte, p, t)
    r["ref"] = dict(ratio=R_REF, thr=t, **c, precision=c["tp"] / max(c["tp"] + c["fp"], 1), recall=c["tp"] / yte.sum(),
                    f1=2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1), cost=R_REF * c["fn"] + c["fp"],
                    cost_never=R_REF * int(yte.sum()))

# bootstrap CI for the winner's test PR-AUC and recall at a 5% FPR budget
rng = np.random.default_rng(SEED); pw = probs_test[winner][0]
tw = res[winner]["fpr_budget"][2]["thr"]; bs_ap, bs_rec = [], []
for _ in range(1000):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].sum() == 0: continue
    bs_ap.append(ap(yte[i], pw[i])); bs_rec.append(((pw[i] >= tw) & (yte[i] == 1)).sum() / yte[i].sum())
boot = dict(ap=[float(x) for x in np.percentile(bs_ap, [2.5, 97.5])], recall5=[float(x) for x in np.percentile(bs_rec, [2.5, 97.5])])

# %% [markdown]
# ## 6. Error analysis for the winner at the reference ratio

# %%
t_ref = res[winner]["ref"]["thr"]; pred = pw >= t_ref
te_an = te.copy(); te_an["p"] = pw; te_an["pred"] = pred
te_an["ftype"] = te_an[TYPES + ["Machine failure"]].apply(
    lambda r: "+".join([c for c in TYPES if r[c] == 1]) or ("unlabelled" if r["Machine failure"] == 1 else "none"), axis=1)
by_type = []
for c in TYPES:
    s = te_an[te_an[c] == 1]; by_type.append(dict(type=c, n=int(len(s)), caught=int(s["pred"].sum())))
s = te_an[(te_an["Machine failure"] == 1) & (te_an[TYPES].sum(axis=1) == 0)]
by_type.append(dict(type="unlabelled", n=int(len(s)), caught=int(s["pred"].sum())))
by_machine = [dict(type=t, n=int(((te_an.Type == t) & (yte == 1)).sum()),
                   caught=int(((te_an.Type == t) & (yte == 1) & te_an.pred).sum())) for t in "LMH"]
cols = ["UDI", "Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "p", "ftype"]
fn_rows = te_an[(yte == 1) & ~te_an.pred].sort_values("p")[cols].round(3).to_dict("records")
fp_rows = te_an[(yte == 0) & te_an.pred].sort_values("p", ascending=False)[cols].round(3).to_dict("records")
print(by_type, by_machine, len(fn_rows), len(fp_rows))

# %% [markdown]
# ## 7. Data-quality facts for the Data section

# %%
eda = dict(n=int(len(raw)), cols=int(raw.shape[1]), missing=int(raw.isna().sum().sum()), failures=int(y_all.sum()),
           rate=float(y_all.mean()), types={t: int(raw[t].sum()) for t in TYPES},
           by_type=[dict(type=t, n=int((raw.Type == t).sum()), fail=int(raw[raw.Type == t]["Machine failure"].sum())) for t in "LMH"],
           fail_no_flag=int(((raw["Machine failure"] == 1) & (raw[TYPES].sum(axis=1) == 0)).sum()),
           flag_no_fail=int(((raw["Machine failure"] == 0) & (raw[TYPES].sum(axis=1) > 0)).sum()),
           multi_flag=int((raw[TYPES].sum(axis=1) > 1).sum()),
           describe=raw[NUM_RAW].describe().round(2).T[["mean", "std", "min", "max"]].reset_index().rename(columns={"index": "name"}).to_dict("records"),
           ntrain=int(len(train)), ntest=int(len(test)), ptrain=int(ytr.sum()), ptest=int(yte.sum()))

# %% [markdown]
# ## 8. Figures

# %%
def save(name): plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close()
fe = add_phys(raw)
# 1 overview
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
nn = [(y_all == 0).sum(), (y_all == 1).sum()]
ax[0].bar(["No failure", "Failure"], nn, color=[CALM, ALARM])
for i, v in enumerate(nn): ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
ax[0].set_title("Rows by outcome"); ax[0].set_ylim(0, 11000)
ax[1].bar(TYPES, [raw[t].sum() for t in TYPES], color=ALARM); ax[1].set_title("Failures by failure type")
for i, t in enumerate(TYPES): ax[1].text(i, raw[t].sum(), int(raw[t].sum()), ha="center", va="bottom")
save("fig_overview")
# 2 physics
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (x, yv, xl, yl) in zip(ax, [("Rotational speed", "Torque", "Rotational speed (rpm)", "Torque (Nm)"),
                                  ("Rotational speed", "Temp diff (K)", "Rotational speed (rpm)", "Process minus air temp (K)"),
                                  ("Tool wear", "Torque", "Tool wear (min)", "Torque (Nm)")]):
    ok, bad = fe[fe["Machine failure"] == 0], fe[fe["Machine failure"] == 1]
    a.scatter(ok[x], ok[yv], s=3, c=CALM, alpha=.25, label="No failure")
    a.scatter(bad[x], bad[yv], s=9, c=ALARM, alpha=.8, label="Failure")
    a.set_xlabel(xl); a.set_ylabel(yl)
ax[0].legend(markerscale=2.5, loc="upper right"); save("fig_physics")
# 3 CV raw vs engineered
fig, ax = plt.subplots(figsize=(7, 3.8)); w = .36
for j, (feat, col, lab) in enumerate([("raw", GREY, "Raw features"), ("eng", INK, "Raw + physics features")]):
    m = [res[f"{k}__{feat}"]["cv_ap_mean"] for k in MODELS]; s = [res[f"{k}__{feat}"]["cv_ap_std"] for k in MODELS]
    ax.bar(np.arange(4) + (j - .5) * w, m, w, yerr=s, capsize=3, color=col, label=lab)
ax.axhline(ytr.mean(), color=ALARM, ls="--", lw=1); ax.text(3.45, ytr.mean() + .015, "majority baseline", color=ALARM, ha="right", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels([v[0] for v in MODELS.values()]); ax.set_ylabel("5-fold CV PR-AUC (mean ± sd)"); ax.legend(fontsize=8, loc="upper left")
save("fig_cv")
# 4 PR curves on test (each model with its better feature set, chosen by CV)
fig, ax = plt.subplots(figsize=(6, 4.2)); cmap = [CALM, "#7A5C9E", ALARM, INK]
for (k, v), c in zip(MODELS.items(), cmap):
    f = max(FEATS, key=lambda ft: res[f"{k}__{ft}"]["cv_ap_mean"]); pr, rc, _ = precision_recall_curve(yte, probs_test[f"{k}__{f}"][0])
    ax.plot(rc, pr, color=c, lw=1.8, label=f"{v[0]} ({f}) AP={res[f'{k}__{f}']['test']['ap']:.2f}")
ax.axhline(yte.mean(), color=GREY, ls="--", lw=1, label="Majority baseline"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=7.5); ax.set_ylim(0, 1.02)
save("fig_pr")
# 5 cost curves relative to "never alarm", winner (test), dots = threshold picked on training folds
fig, ax = plt.subplots(figsize=(7, 4)); fnt, fpt = fn_fp_grid(yte, pw)
for r, c in zip([2, 5, 10, 50, 100], ["#B8C4D6", GREY, ALARM, CALM, INK]):
    rel = (r * fnt + fpt) / (r * yte.sum()); ax.plot(TGRID, rel, color=c, lw=1.8, label=f"cost ratio {r}:1")
    t = res[winner]["thr_raw"][int(np.argmin(abs(RATIOS - r)))]; ax.plot([t], [rel[int(round(t * 1000))]], "o", color=c, ms=6)
ax.axhline(1, color=GREY, ls=":", lw=1); ax.set_ylim(0, 2.2); ax.set_xlabel("Alarm threshold"); ax.set_ylabel("Test cost ÷ cost of never alarming")
ax.legend(fontsize=8); ax.set_title(f"{res[winner]['label']}: dots = threshold chosen on training folds"); save("fig_cost")
# 6 optimal threshold vs ratio
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for (k, v), c in zip(MODELS.items(), cmap):
    f = max(FEATS, key=lambda ft: res[f"{k}__{ft}"]["cv_ap_mean"]); ax.plot(RATIOS, res[f"{k}__{f}"]["thr_raw"], color=c, lw=1.8, label=v[0])
ax.set_xscale("log"); ax.set_xlabel("Cost of missed failure ÷ cost of false alarm"); ax.set_ylabel("Best threshold (training folds)"); ax.legend(fontsize=8); save("fig_threshold")
# 7 reliability
fig, ax = plt.subplots(figsize=(4.8, 4.4)); ax.plot([0, 1], [0, 1], ":", color=GREY)
for pp, lab, c in [(probs_test[winner][0], "Raw (class-weighted)", ALARM), (probs_test[winner][1], f"Calibrated ({res[winner]['cal_method']})", CALM)]:
    fy, mp = calibration_curve(yte, pp, n_bins=6, strategy="quantile"); ax.plot(mp, fy, "o-", color=c, label=lab)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed failure rate"); ax.legend(fontsize=8); save("fig_calibration")
# 8 recall by failure type
fig, ax = plt.subplots(figsize=(6.5, 3.6)); lab = [f"{d['type']}\n(n={d['n']})" for d in by_type]
sh = [d["caught"] / max(d["n"], 1) for d in by_type]
ax.bar(lab, sh, color=[CALM if s >= .5 else ALARM for s in sh])
ax.set_ylim(0, 1.05); ax.set_ylabel("Share caught (test set)"); ax.set_title(f"Recall by failure type at ratio {R_REF:g}:1"); save("fig_errors")

# %% [markdown]
# ## 9. Export everything the website needs

# %%
test_rows = [[float(r["Air temperature"]), float(r["Process temperature"]), int(r["Rotational speed"]), float(r["Torque"]),
              int(r["Tool wear"]), "LMH".index(r["Type"]), r["ftype"]] for _, r in te_an.iterrows()]
out = dict(meta=dict(seed=SEED, ratios=RATIOS.tolist(), r_ref=R_REF, winner=winner, fpr_budgets=FPR_BUDGETS, boot=boot,
                     baseline=baseline, features_raw=NUM_RAW, features_eng=NUM_ENG[5:]),
           eda=eda, models=res,
           errors=dict(by_type=by_type, by_machine=by_machine, fn=fn_rows, fp=fp_rows[:12], n_fp=len(fp_rows), thr=t_ref),
           test=dict(y=yte.tolist(), udi=te["UDI"].tolist(), rows=test_rows,
                     raw={k: np.round(v[0], 4).tolist() for k, v in probs_test.items()},
                     cal={k: np.round(v[1], 4).tolist() for k, v in probs_test.items()}))
(ROOT / "site_data" / "results.json").write_text(json.dumps(out, separators=(",", ":")))
print("done; json KB:", round((ROOT / "site_data" / "results.json").stat().st_size / 1024))
