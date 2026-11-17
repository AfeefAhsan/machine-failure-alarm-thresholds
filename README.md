# Choosing Alarm Thresholds for Machine Failure Prediction

ES 335 course project. Afeef Ahsan (24110017) and Shared Nada T (24110329).

**Question.** Given a milling machine's operating readings, how does the decision threshold change the balance between missed failures and false alarms, and how does the best threshold shift as the cost of a missed failure (relative to a false alarm) changes?

**Live site:** https://afeefahsan.github.io/machine-failure-alarm-thresholds/

**Repository:** https://github.com/AfeefAhsan/machine-failure-alarm-thresholds

## What is in this repository

| Path | Contents |
|---|---|
| `index.html` | The finished website: one self-contained file (CSS, JS, data and figures inlined). Open it directly or publish it. |
| `site_src/` | Editable sources of the site (`template.html`, `style.css`, `app.js`, `explore.js`). Edit these, then rebuild. |
| `site_data/` | `results.json` (precomputed test probabilities, thresholds, metrics) and `tree.json` (the small tree used by the lab). |
| `figures/` | Figures used on the site, written by the pipeline. |
| `src/build_site.py`, `src/export_tree.py` | Export the lab tree and assemble `index.html` from `site_src/`. |
| `src/pipeline.py` | The whole analysis: split, preprocessing, models, thresholds, calibration, figures, JSON export. |
| `notebooks/analysis.ipynb` | The same code as a notebook (cells match `pipeline.py`; outputs are not stored, run it to see them). |
| `data/raw/ai4i2020.csv` | The original dataset. |
| `data/train.csv`, `data/test.csv` | The exact 80/20 stratified split (seed 42): 8,000 training rows (271 failures), 2,000 test rows (68 failures). |

## Data and access

AI4I 2020 Predictive Maintenance Dataset, S. Matzka, UCI Machine Learning Repository, doi:10.24432/C5HS5C. Public and free; the UCI page lists the CC BY 4.0 licence (check it once more before submitting). One row is one simulated operating snapshot. The data is synthetic, so results say how methods behave under class imbalance, not how real machines fail. No credentials or private data are used.

## Reproduce

```bash
pip install -r requirements.txt
python src/pipeline.py          # about 5 minutes on one CPU core
python src/export_tree.py       # small tree for the website lab
python src/build_site.py        # writes the self-contained index.html
```

`index.html` needs nothing else: double-click it, or publish it.

Details that fix the result: split is `train_test_split(test_size=0.2, stratify=failure, random_state=42)`; cross-validation is 5-fold stratified with shuffle and seed 42; scaling and encoding sit inside scikit-learn `Pipeline`s; the failure-type columns and IDs are never model inputs; thresholds are chosen on out-of-fold training predictions only; the test set is used once for reporting.

## Publish on GitHub Pages

1. Create a public repository and push everything in this folder (keep `.nojekyll`).
2. Settings → Pages → Build from a branch → `main` / root.
3. The site is served at https://afeefahsan.github.io/machine-failure-alarm-thresholds/

## AI-use disclosure

Claude (Anthropic) was used to assist with writing the code, debugging, and implementing the website. The project ideas, overall approach, analysis decisions, and visualisation concepts were developed by our team. We used Claude as a coding assistant to bring our ideas into implementation. We ran and tested the code, verified the results, and take responsibility for the final work.

## Team contributions

Afeef Ahsan worked on developing and testing the models, creating physics-based features, and designing the interactive machine prediction section. Shared Nada T worked on comparing model results, analysing missed failures and false alarms, and studying how different alarm thresholds and failure costs affect predictions. Both contributed to the project ideas, visualisation design, and interpretation of results.

## Citations

Matzka, S. (2020). Explainable artificial intelligence for predictive maintenance applications. *Third International Conference on Artificial Intelligence for Industries (AI4I)*, 69-74. Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, doi:10.24432/C5HS5C. Libraries: scikit-learn, pandas, NumPy, Matplotlib, Seaborn.
