
# DSA 210 - PM2.5 Air Pollution, Life Expectancy & HDI Analysis

## Research Question

Is higher PM2.5 exposure associated with lower life expectancy, and does the
Human Development Index (HDI) moderate this relationship?

---


## Data Sources

The soures of PM2.5 exposure, Life expectancy and Human Development Index are listed below. All data is filtered to the year **2020** inside the script. The data is also shared within the repository.

| Variable | Source | File to download |
|---|---|---|
| PM2.5 exposure (µg/m³) | [World Bank](https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3) | `Data.csv` |
| Life expectancy (years) | [Our World in Data](https://ourworldindata.org/life-expectancy) | `life-expectancy-hmd-unwpp.csv` |
| Human Development Index | [Our World in Data](https://ourworldindata.org/human-development-index) | `human-development-index.csv` |

### How countries were matched across datasets

Each dataset covers a different number of countries (PM2.5: 200, Life Expectancy: 201, HDI: 192).
The three files are merged using an **inner join** on country name, meaning only countries
present in **all three** datasets are kept. This results in **166 countries** in the final
dataset. Countries missing from even one source are excluded to avoid NaN values in the analysis.

---

## Requirements

- Python 3.10 or higher
- The five libraries listed in `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
scipy
```

---

## How to Reproduce

### Option A – Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Upload `analysis.py` and the three CSV data files using the file panel on the left.
3. Install dependencies (most are pre-installed in Colab, but run this to be safe):
   ```python
   !pip install -r requirements.txt
   ```
4. Upload three files in Google Colab to the "Files" section:
   ```python
   
   pd.read_csv("/content/2f46c15c-49bc-4ef9-807f-78b4b0075a28_Data.csv")
   pd.read_csv("/content/life-expectancy-hmd-unwpp.csv")
   pd.read_csv("/content/human-development-index.csv")

5. Run the script in order from top to bottom, do not skip cells as later
   steps depend on variables created in earlier ones (e.g. `PM25_Cat` and
   `HDI_Group` must be created in Step 3 before Step 5 can run)

---

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
5. Run the script

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

### Hypothesis tests (printed to console)

**Primary hypothesis:**
- H0: Higher PM2.5 exposure is not associated with lower life expectancy
- H1: Higher PM2.5 exposure is associated with lower life expectancy

| ID | Test | Purpose |
|---|---|---|
| H1 | One-way ANOVA + post-hoc pairwise t-tests | Mean life expectancy across 4 HDI groups |
| H2 | Two-sample t-test | Mean LE between Low PM2.5 vs High PM2.5 countries |
| H3 | Two-sample t-test | Mean LE between Low HDI vs Very High HDI countries |
| H4 | Chi-square test | Association between PM2.5 category and HDI tier |

**Note on paired t-test:** not used because it requires two measurements on the
same subject (e.g. before/after). This dataset has one cross-sectional
observation per country for 2020 only.

Significance level: **alpha = 0.05** throughout. This is the conventional threshold
used in social and health sciences, meaning we accept a 5% chance of a false
positive. Bonferroni correction applied to all post-hoc pairwise comparisons.

---

## Variables

| Variable | Type | Unit | Role |
|---|---|---|---|
| PM2.5 | Continuous | µg/m³ | Independent variable |
| Life Expectancy | Continuous | Years | Dependent variable |
| HDI | Continuous | 0–1 index | Moderating variable |
| PM25_Cat | Categorical | — | PM2.5 split into Low (<=15) / High (>15) using WHO threshold |
| HDI_Group | Categorical | — | HDI split into 4 tiers using UNDP thresholds |
| Country | Categorical | — | Identifier |

### HDI group thresholds (UNDP standard)
| Group | HDI Range |
|---|---|
| Low | < 0.55 |
| Medium | 0.55 – 0.70 |
| High | 0.70 – 0.80 |
| Very High | >= 0.80 |

### PM2.5 category threshold
Countries are split at **15 µg/m³**, which was the WHO interim annual target
in use during the 2020 reference period of this data.

### Chi-square validity check
All expected cell frequencies must be >= 5 for the chi-square test to be
mathematically valid. Values below 5 would inflate the test statistic and
risk a false positive. The minimum expected frequency in this analysis was
**8.82**, confirming the test is valid.
