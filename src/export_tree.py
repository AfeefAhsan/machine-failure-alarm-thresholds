"""Train a small, readable decision tree on the TRAINING split and export it for the website's 'build a machine' lab.
Leaf probabilities are plain failure rates among training rows in each leaf (honest, unweighted)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
tr, te = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/test.csv")
FEATS = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Power (W)", "Temp gap (K)", "Wear x torque"]
def X(d):
    o = pd.DataFrame({"Type": d["Type"].map({"L": 0, "M": 1, "H": 2})})
    for c in ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]: o[c] = d[c]
    o["Power (W)"] = d["Torque"] * d["Rotational speed"] * 2 * np.pi / 60
    o["Temp gap (K)"] = d["Process temperature"] - d["Air temperature"]
    o["Wear x torque"] = d["Tool wear"] * d["Torque"]
    return o[FEATS]
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42).fit(X(tr), tr["Machine failure"])
t = clf.tree_; val = t.value[:, 0, :]; val = val / val.sum(axis=1, keepdims=True)
ap = float(average_precision_score(te["Machine failure"], clf.predict_proba(X(te))[:, 1]))
out = dict(feats=FEATS, left=t.children_left.tolist(), right=t.children_right.tolist(), feat=t.feature.tolist(),
           thr=[round(float(x), 3) for x in t.threshold], p=[round(float(x), 4) for x in val[:, 1]], n=t.n_node_samples.tolist(), test_ap=round(ap, 3))
(ROOT / "site_data/tree.json").write_text(json.dumps(out, separators=(",", ":")))
print("leaves", int((t.children_left == -1).sum()), "test AP", ap)
