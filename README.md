# DSA 210 Project — PM2.5 Air Pollution, Life Expectancy & HDI Analysis

## Research Question

Is higher PM2.5 exposure associated with lower life expectancy, and does the
Human Development Index (HDI) moderate this relationship?

🌍 **Web app:** https://dsa-210-project-2rx7m23jppcgam23nhxr4w.streamlit.app

---

## Data Sources

The sources of PM2.5 exposure, Life Expectancy and Human Development Index
are listed below. All data is filtered to the year 2020 inside the script.
The data is also shared within the repository.

| Variable | Source | File to download |
|---|---|---|
| PM2.5 exposure (µg/m³) | World Bank | `Data.csv` |
| Life expectancy (years) | Our World in Data | `life-expectancy-hmd-unwpp.csv` |
| Human Development Index | Our World in Data | `human-development-index.csv` |

### How countries were matched across datasets

Each dataset covers a different number of countries (PM2.5: 200, Life
Expectancy: 201, HDI: 192). The three files are merged using an **inner join
on country name**, meaning only countries present in all three datasets are
kept. This results in **190 countries** in the final dataset. Countries
missing from even one source are excluded to avoid NaN values in the analysis.

---

## Requirements

- Python 3.10 or higher
- The libraries listed in `requirements.txt`:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `scikit-learn` *(for the machine-learning section)*
  - `streamlit` *(for the interactive web app only)*
  - `plotly` *(for the interactive web app only)*

---

## How to Reproduce

### Option A – Google Colab (recommended)

1. Open Google Colab and create a new notebook.
2. Upload `analysis.py` and the three CSV data files using the file panel
   on the left.
3. Install dependencies (most are pre-installed in Colab, but run this to be safe):
   ```bash
   !pip install -r requirements.txt
   ```
4. Upload three files in Google Colab to the "Files" section:
   ```python
   pd.read_csv("/content/2f46c15c-49bc-4ef9-807f-78b4b0075a28_Data.csv")
   pd.read_csv("/content/life-expectancy-hmd-unwpp.csv")
   pd.read_csv("/content/human-development-index.csv")
   ```
5. Run the script in order from top to bottom — do not skip cells, as later
   steps depend on variables created in earlier ones (e.g. `PM25_Cat` and
   `HDI_Group` must be created in Step 3 before Step 5 can run).

### Option B – Local Machine

1. Clone or download this project folder.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Update the three file paths in `analysis.py` to point to your CSV files:
   ```python
   pd.read_csv("Data.csv")
   pd.read_csv("life-expectancy-hmd-unwpp.csv")
   pd.read_csv("human-development-index.csv")
   ```
5. Run the script.

---

## What the Script Produces

### EDA outputs (printed to console)

- Descriptive statistics (mean, std, min, max, skewness, kurtosis)
- Missing value counts
- IQR outlier detection for PM2.5 and Life Expectancy
- Country counts per HDI group and PM2.5 category

### Figures (saved as PNG)

| File | Description |
|---|---|
| `figure1_distributions.png` | Histograms + KDE for PM2.5, Life Expectancy, and HDI |
| `figure2_boxplots_by_hdi_group.png` | Box plots of PM2.5 and Life Expectancy by HDI group |
| `figure3_scatter_pairs.png` | Pairwise scatter plots with Pearson r and Spearman rho |
| `figure4_heatmap_regional.png` | Correlation heat-map and regional scatter |
| `figure5_scatter_hdi_group.png` | PM2.5 vs Life Expectancy coloured by HDI group |
| `figure6_le_by_pm25_category.png` | Life Expectancy by PM2.5 category (WHO threshold) |

---

## Hypothesis Tests (printed to console)

**Primary hypothesis:**

> **H0:** Higher PM2.5 exposure is *not* associated with lower life expectancy
>
> **H1:** Higher PM2.5 exposure *is* associated with lower life expectancy

| ID | Test | Purpose |
|---|---|---|
| H1 | One-way ANOVA + post-hoc pairwise t-tests | Mean life expectancy across 4 HDI groups |
| H2 | Two-sample t-test | Mean LE between Low PM2.5 vs High PM2.5 countries |
| H3 | Two-sample t-test | Mean LE between Low HDI vs Very High HDI countries |
| H4 | Chi-square test | Association between PM2.5 category and HDI tier |

**Note on paired t-test:** not used because it requires two measurements on
the same subject (e.g. before/after). This dataset has one cross-sectional
observation per country for 2020 only.

**Significance level:** α = 0.05 throughout. This is the conventional
threshold used in social and health sciences, meaning we accept a 5 % chance
of a false positive. Bonferroni correction applied to all post-hoc pairwise
comparisons.

---

## Machine Learning Analysis

Four methods are applied in sequence, each chosen because it matches the data.

### Linear regression (three nested models)

```
M1: LifeExp ~ PM2.5                                  the "raw" effect
M2: LifeExp ~ PM2.5 + HDI                            controls for HDI (confounding test)
M3: LifeExp ~ PM2.5 + HDI + (PM2.5 × HDI)            formal moderation test (mean-centred)
```

For each model: coefficients with 95 % CIs, R², adjusted R², RMSE, and 5-fold
cross-validated R² are reported. M1 → M2 shows whether HDI is a confounder;
the interaction term in M3 directly tests the moderation hypothesis.

### Random Forest regression

Predicts LifeExp from PM2.5 + HDI with 500 trees and 5-fold cross-validation.
Permutation feature importance ranks PM2.5 vs HDI in a non-parametric way
and corroborates (or challenges) the linear regression conclusions.

### K-Means clustering

Standardised features `[PM2.5, HDI, LifeExp]`; k = 2…8 is swept and the
optimal k is chosen by silhouette score. The resulting clusters are
cross-tabulated against `HDI_Group` and `Region` to interpret what each
cluster represents.

### PCA (dimensionality reduction)

The 3 standardised features are projected into 2 principal components. PCA
loadings explain what each axis represents (PC1 = "development axis", PC2 =
residual PM2.5). The PCA scatter is colored by HDI group, by K-Means
cluster, and by world region.

### Performance metrics produced

| Model | Metric |
|---|---|
| Linear regression M1 / M2 / M3 | R², Adj. R², RMSE (years), 5-fold CV R² |
| Random Forest | 5-fold CV R², 5-fold CV RMSE, permutation feature importance |
| K-Means | Silhouette score, inertia (elbow plot), cluster size, cluster × HDI / cluster × Region tables |
| PCA | Explained variance ratio per PC, total variance captured, component loadings |

> **Why XGBoost / logistic regression / hierarchical clustering were not used:**
> XGBoost is overkill for n = 190 with 2 predictors and would simply overfit.
> Logistic regression was considered but the outcome (life expectancy in
> years) is continuous, so binarising it would discard information.
> Hierarchical clustering yields effectively the same country groupings as
> K-Means in this dataset.

### Sensitivity analysis (Gulf states)

The Gulf states (Qatar, Bahrain, Kuwait, Saudi Arabia, Oman, UAE) have very
high PM2.5 driven by natural desert dust rather than combustion. All
correlation and regression tests are re-run with these 6 countries excluded
to check whether the conclusions are robust.

---

## Variables

| Variable | Type | Unit | Role |
|---|---|---|---|
| `PM2.5` | Continuous | µg/m³ | Independent variable |
| `Life Expectancy` | Continuous | Years | Dependent variable |
| `HDI` | Continuous | 0–1 index | Moderating variable |
| `PM25_Cat` | Categorical | — | PM2.5 split into Low (≤15) / High (>15) using WHO threshold |
| `HDI_Group` | Categorical | — | HDI split into 4 tiers using UNDP thresholds |
| `Country` | Categorical | — | Identifier |

### HDI group thresholds (UNDP standard)

| Group | HDI Range |
|---|---|
| Low | < 0.55 |
| Medium | 0.55 – 0.70 |
| High | 0.70 – 0.80 |
| Very High | ≥ 0.80 |

### PM2.5 category threshold

Countries are split at **15 µg/m³**, which was the WHO interim annual target
in use during the 2020 reference period of this data.

### Chi-square validity check

All expected cell frequencies must be ≥ 5 for the chi-square test to be
mathematically valid. Values below 5 would inflate the test statistic and
risk a false positive. The minimum expected frequency in this analysis was
**8.82**, confirming the test is valid.

---

## Interactive Web App

The deployed app is at https://dsa-210-project-2rx7m23jppcgam23nhxr4w.streamlit.app
— **no install required, just open the link in any browser.** The page
reproduces the entire analysis with live filters: the user can toggle Gulf
states on/off, choose which world regions to include, and choose which HDI
groups to include, and every test, correlation, regression coefficient and
chart on the page recomputes in real time.

**Six tabs:**

- **The data** — distributions and box plots, all live-filtered
- **Relationships** — correlation heat-maps and an interactive scatter
- **Hypothesis tests** — Pearson, Spearman, ANOVA, pairwise t-tests
- **Regression** — M1 / M2 / M3 with live coefficient comparison
- **Machine learning** — Random Forest, K-Means, PCA
- **Country explorer** — look up any country and see where it sits

### *(Optional — for reviewers who want to run the app from source)*

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). The
file `merged_dataset_2020.csv` must be in the same folder as
`streamlit_app.py`.

---

## Deploying the App Online (Streamlit Community Cloud)

1. Push `streamlit_app.py`, `requirements.txt` and `merged_dataset_2020.csv`
   to a public GitHub repository.
2. Go to https://share.streamlit.io.
3. Click **New app**, paste the repository URL, set the main file to
   `streamlit_app.py` and click **Deploy**.
4. The site goes live within ~2 minutes at
   `https://<app-name>.streamlit.app`.

---

## Key Findings

- **PM2.5 and life expectancy are negatively correlated globally** — Pearson
  r ≈ −0.48, p ≈ 3 × 10⁻¹².
- **HDI dominates life expectancy** — Pearson r ≈ +0.91. M2 R² jumps from
  0.23 (PM2.5 only) to 0.82 once HDI is added.
- **The PM2.5 effect vanishes when HDI is controlled for** — its coefficient
  drops from −0.23 (p ≈ 10⁻¹²) to +0.02 (p ≈ 0.19).
- **No statistical evidence of moderation by HDI** — the interaction term in
  M3 has p ≈ 0.69.
- **Random Forest agrees** — permutation importance puts HDI roughly 23×
  above PM2.5.
- **Countries fall into 2 natural clusters** — silhouette peaks at k = 2
  ("developed/clean/long-lived" vs "developing/polluted/shorter-lived").
- **2 principal components explain ≈ 97 % of the variance** — PC1 =
  development axis, PC2 = residual PM2.5.

**Overall:** the apparent global link between PM2.5 and life expectancy is
largely a **confounding effect of HDI**, not a moderating one. This
contradicts the project's original moderation hypothesis but is itself a
clean, multi-method finding.

---

## Limitations

- **Single year (2020).** The 2020 sample is also affected by COVID-era
  mortality, especially in developed countries.
- **Country-level aggregation.** Within-country variation (urban vs rural,
  income groups) is invisible to this analysis.
- **Observational data.** The results show *associations* and pinpoint
  *confounding*; they do not by themselves prove a causal effect of PM2.5
  on life expectancy.
- **HDI mechanically contains life expectancy.** HDI is built from three
  components, one of which is life expectancy itself, so part of the strong
  HDI ↔ LifeExp correlation is structural rather than substantive.
- **Gulf states.** Their PM2.5 is largely natural desert dust with a
  different health profile than combustion PM2.5; the sensitivity analysis
  addresses this.
