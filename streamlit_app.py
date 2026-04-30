# =============================================================
# DSA 210 - Interactive Project Page
# Air Pollution (PM2.5), Life Expectancy & HDI - Year 2020
# =============================================================
# Run locally:
#     streamlit run streamlit_app.py
# Deploy free:
#     Push to GitHub + go to share.streamlit.io
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.inspection      import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score

# =============================================================
# PAGE SETUP
# =============================================================
st.set_page_config(
    page_title="PM2.5, HDI & Life Expectancy",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the slick blog-post feel
st.markdown("""
<style>
    /* Tighter top padding */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

    /* Big punchy headers */
    h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
    h2 { font-weight: 700 !important; margin-top: 2rem !important;
         border-bottom: 2px solid #f0f2f6; padding-bottom: 0.3rem;}
    h3 { font-weight: 600 !important; }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; opacity: 0.7; }

    /* Pretty quote / callout block */
    .callout {
        background: linear-gradient(135deg, #f0f9ff 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-size: 1.02rem;
        !important;
    }
    .callout_2 {
        background: linear-gradient(135deg, #f3f4e8 0%, #e0e3c8 100%);
        border-left: 5px solid #84a02b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        !important;

    }
    .callout-blue {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        !important;
    }
    .callout-green {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        !important;
    }
    .callout-red {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        !important;
    }

    /* Tab styling - bigger, bolder */
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 1rem; padding: 0.75rem 1.5rem;
    }

    /* Subtle hover on dataframes */
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# DATA LOADING (cached)
# =============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_dataset_2020.csv")
    df = df.rename(columns={"PM2.5": "PM2_5"})
    return df


df_full = load_data()

GULF_CODES   = ["QAT", "BHR", "KWT", "ARE", "SAU", "OMN"]
GULF_NAMES   = "Qatar, Bahrain, Kuwait, Saudi Arabia, Oman, UAE"
HDI_ORDER    = ["Low", "Medium", "High", "Very High"]
HDI_PALETTE  = {"Low": "#ef4444", "Medium": "#f59e0b",
                "High": "#3b82f6", "Very High": "#10b981"}


# =============================================================
# SIDEBAR - filters
# =============================================================
with st.sidebar:
    st.markdown("## Controls")
    st.caption("Every chart, test, and metric on the page recomputes "
               "live from these filters.")

    st.markdown("### Gulf states")
    exclude_gulf = st.toggle(
        "Exclude Gulf countries",
        value=False,
        help=(f"Toggles the 6 GCC countries: {GULF_NAMES}. Their PM2.5 "
              "comes mostly from natural desert dust, not combustion. "
              "Watch the within-Very-High-HDI correlation flip when "
              "you turn this on!")
    )

    st.markdown("### Regions")
    regions = sorted(df_full["Region"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "Include regions:", regions, default=regions
    )

    st.markdown("### HDI groups")
    selected_hdi = st.multiselect(
        "Include HDI groups:",
        HDI_ORDER, default=HDI_ORDER
    )

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "**Data sources** — World Bank (PM2.5), UNDP / Our World in "
        "Data (HDI, life expectancy). All values are for 2020.\n\n"
        "**Methods** — descriptive EDA, correlation tests, ANOVA, "
        "linear regression, Random Forest, K-Means, PCA.\n\n"
        "**Sample size** — 190 countries (or 184 with Gulf excluded)."
    )

# Apply filters
df = df_full.copy()
if exclude_gulf:
    df = df[~df["Code"].isin(GULF_CODES)]
df = df[df["Region"].isin(selected_regions)]
df = df[df["HDI_Group"].isin(selected_hdi)]
df = df.reset_index(drop=True)

if len(df) < 10:
    st.error(" Too few countries selected — pick more regions or HDI groups.")
    st.stop()


# =============================================================
# HEADER
# =============================================================
st.markdown("#  Does air pollution actually shorten lives?")
st.markdown(
    "### Or does development do all the work? &nbsp;"
    "<span style='color:#888;font-weight:400;'>"
    "An interactive look at PM2.5, HDI and life expectancy across "
    f"{len(df)} countries in 2020.</span>",
    unsafe_allow_html=True
)

# Live metric strip
col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries", f"{len(df)}")
col2.metric("Mean PM2.5", f"{df['PM2_5'].mean():.1f} µg/m³")
col3.metric("Mean HDI", f"{df['HDI'].mean():.3f}")
col4.metric("Mean Life Exp.", f"{df['LifeExp'].mean():.1f} yrs")

st.markdown("""
<div class="callout_2">
</b> Air pollution is an important problem the world is facing today that can lead to severe 
health consequences. In this project I wanted to invesitgate the relationship between air pollution by the means of 
PM2.5( particules that have a diamter smaller than 2.5 micrometers) and life expectation. I also wanted to see whether a 
better HDI ( Human Development Index) can prevent effects of air pollution.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callout">
This page works through that
question with EDA, hypothesis tests, regression, and a few ML methods,
and you can re&#8209; run everything live by toggling the controls in the
sidebar.
</div>
""", unsafe_allow_html=True)


# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " The data",
    " Relationships",
    " Hypothesis tests",
    " Regression",
    " Machine learning",
    " Country explorer",
])


# -------------------------------------------------------------
# TAB 1 - THE DATA
# -------------------------------------------------------------
with tab1:
    st.markdown("## How is each variable spread?")
    st.caption("Histograms (with KDE) above, boxplots below. Hover any "
               "bar or point for details.")

    c1, c2, c3 = st.columns(3)

    fig = px.histogram(df, x="PM2_5", nbins=30, marginal="box",
                       color_discrete_sequence=["#3b82f6"])
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                      title="PM2.5 (µg/m³)", showlegend=False)
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="HDI", nbins=30, marginal="box",
                       color_discrete_sequence=["#10b981"])
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                      title="HDI (0–1)", showlegend=False)
    c2.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="LifeExp", nbins=30, marginal="box",
                       color_discrete_sequence=["#f59e0b"])
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                      title="Life Expectancy (years)", showlegend=False)
    c3.plotly_chart(fig, use_container_width=True)

    st.markdown("## By HDI group")
    st.caption("How does each variable change across the four UNDP HDI tiers?")

    c1, c2 = st.columns(2)
    fig = px.box(df, x="HDI_Group", y="PM2_5",
                 category_orders={"HDI_Group": HDI_ORDER},
                 color="HDI_Group", color_discrete_map=HDI_PALETTE,
                 points="all", hover_data=["Country"])
    fig.update_layout(height=400, showlegend=False,
                      title="PM2.5 by HDI group")
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="HDI_Group", y="LifeExp",
                 category_orders={"HDI_Group": HDI_ORDER},
                 color="HDI_Group", color_discrete_map=HDI_PALETTE,
                 points="all", hover_data=["Country"])
    fig.update_layout(height=400, showlegend=False,
                      title="Life Expectancy by HDI group")
    c2.plotly_chart(fig, use_container_width=True)

    with st.expander(" Numeric summary"):
        st.dataframe(
            df[["PM2_5", "HDI", "LifeExp"]].describe().round(3),
            use_container_width=True
        )


# -------------------------------------------------------------
# TAB 2 - RELATIONSHIPS
# -------------------------------------------------------------
with tab2:
    st.markdown("## Correlation heatmap")

    pearson  = df[["PM2_5", "HDI", "LifeExp"]].corr()
    spearman = df[["PM2_5", "HDI", "LifeExp"]].corr(method="spearman")

    c1, c2 = st.columns(2)
    fig = px.imshow(pearson, text_auto=".3f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="equal")
    fig.update_layout(title="Pearson", height=380, coloraxis_showscale=False)
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.imshow(spearman, text_auto=".3f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="equal")
    fig.update_layout(title="Spearman", height=380, coloraxis_showscale=False)
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown("## Interactive scatter")
    st.caption("Hover any point for the country. Click HDI groups in the "
               "legend to isolate them.")

    c1, c2 = st.columns([1, 3])
    x_var = c1.selectbox("X axis", ["PM2_5", "HDI", "LifeExp"], index=0)
    y_var = c1.selectbox("Y axis", ["PM2_5", "HDI", "LifeExp"], index=2)
    show_trend = c1.checkbox("Overall trend line", value=True)
    color_by = c1.radio("Color by", ["HDI_Group", "Region"], index=0)

    color_map = HDI_PALETTE if color_by == "HDI_Group" else None
    cat_orders = ({"HDI_Group": HDI_ORDER} if color_by == "HDI_Group"
                  else None)

    fig = px.scatter(
        df, x=x_var, y=y_var, color=color_by,
        color_discrete_map=color_map, category_orders=cat_orders,
        hover_data=["Country", "Region", "HDI", "PM2_5", "LifeExp"],
        trendline="ols" if show_trend else None,
        trendline_scope="overall" if show_trend else None,
        trendline_color_override="#111827" if show_trend else None,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    c2.plotly_chart(fig, use_container_width=True)

    # Quick correlation card
    if x_var != y_var:
        r, p_p = stats.pearsonr(df[x_var], df[y_var])
        rho, p_s = stats.spearmanr(df[x_var], df[y_var])
        st.markdown(f"""
<div class="callout-blue">
For the variables and filters you selected:<br>
<b>Pearson r</b> = {r:+.3f} (p = {p_p:.2e}) &nbsp;·&nbsp;
<b>Spearman ρ</b> = {rho:+.3f} (p = {p_s:.2e})
</div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 3 - HYPOTHESIS TESTS
# -------------------------------------------------------------
with tab3:
    st.markdown("## Hypothesis tests &nbsp;<small style='color:#888;font-weight:400;'>α = 0.05</small>",
                unsafe_allow_html=True)
    st.caption("All p-values are recomputed live from the current filter selection.")

    def verdict(p, a=0.05):
        return " Reject H₀" if p < a else " Fail to reject H₀"

    # Correlation tests
    st.markdown("### Correlation tests")
    rows = []
    for x, y_ in [("PM2_5","LifeExp"), ("HDI","LifeExp"), ("PM2_5","HDI")]:
        r, p_p = stats.pearsonr(df[x], df[y_])
        rho, p_s = stats.spearmanr(df[x], df[y_])
        rows.append({
            "Pair": f"{x} vs {y_}",
            "Pearson r": f"{r:+.4f}",
            "Pearson p": f"{p_p:.3e}",
            "Pearson decision": verdict(p_p),
            "Spearman ρ": f"{rho:+.4f}",
            "Spearman p": f"{p_s:.3e}",
            "Spearman decision": verdict(p_s),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ANOVA
    st.markdown("### ANOVA across HDI groups")
    if len(df["HDI_Group"].unique()) >= 2:
        rows = []
        for col in ["LifeExp", "PM2_5"]:
            groups = [df[df["HDI_Group"]==g][col].values
                      for g in HDI_ORDER if g in df["HDI_Group"].unique()]
            if len(groups) >= 2 and all(len(g)>=2 for g in groups):
                F, p_a = stats.f_oneway(*groups)
                rows.append({"Variable": col, "F": f"{F:.3f}",
                             "p-value": f"{p_a:.3e}", "Decision": verdict(p_a)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 HDI groups with ≥2 countries each.")
    else:
        st.info("Select more HDI groups to run ANOVA.")

    # Pairwise t-tests
    st.markdown("### Pairwise Student's t-tests")
    from itertools import combinations
    available_groups = [g for g in HDI_ORDER if g in df["HDI_Group"].unique()
                         and len(df[df["HDI_Group"]==g]) >= 2]
    if len(available_groups) >= 2:
        target = st.radio("Variable:", ["LifeExp", "PM2_5"], horizontal=True)
        rows = []
        for g1, g2 in combinations(available_groups, 2):
            a = df[df["HDI_Group"]==g1][target]
            b = df[df["HDI_Group"]==g2][target]
            t, p_t = stats.ttest_ind(a, b, equal_var=True)
            rows.append({"Group A": g1, "Group B": g2,
                         "t": f"{t:+.3f}", "p-value": f"{p_t:.3e}",
                         "Decision": verdict(p_t)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
<div class="callout-blue">
<b>How to read this:</b> Pearson and Spearman ask whether two variables
move together. ANOVA asks whether HDI groups have different mean values.
Pairwise t-tests then identify <em>which</em> groups differ. Try toggling
the Gulf-states filter and watch a few of these results change.
</div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 4 - REGRESSION
# -------------------------------------------------------------
with tab4:
    st.markdown("## Linear regression: M1 → M2 → M3")
    st.caption("Three nested models that directly test the project's "
               "moderation hypothesis.")

    def ols_inference(X, y, names):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n = len(y); X_ = np.column_stack([np.ones(n), X])
        p = X_.shape[1]; df_resid = n - p
        XtX_inv = np.linalg.inv(X_.T @ X_)
        beta = XtX_inv @ X_.T @ y
        resid = y - X_ @ beta
        sse = (resid**2).sum(); sigma2 = sse / df_resid
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        t = beta / se
        pval = 2 * (1 - stats.t.cdf(np.abs(t), df=df_resid))
        sst = ((y - y.mean())**2).sum()
        r2 = 1 - sse/sst
        adj = 1 - (1-r2)*(n-1)/(n-p)
        return pd.DataFrame({"feature":["Intercept"]+names,
                             "coef":beta,"std_err":se,"t":t,"p":pval}), \
               {"R2":r2, "Adj_R2":adj, "RMSE":np.sqrt(sse/n), "n":n}

    y = df["LifeExp"].values

    # M1
    X1 = df[["PM2_5"]].values
    t1, m1 = ols_inference(X1, y, ["PM2_5"])

    # M2
    X2 = df[["PM2_5", "HDI"]].values
    t2, m2 = ols_inference(X2, y, ["PM2_5", "HDI"])

    # M3 (mean-centered)
    pm_c  = df["PM2_5"] - df["PM2_5"].mean()
    hdi_c = df["HDI"] - df["HDI"].mean()
    X3 = np.column_stack([pm_c, hdi_c, pm_c*hdi_c])
    t3, m3 = ols_inference(X3, y, ["PM2_5_c", "HDI_c", "PM2.5_c × HDI_c"])

    c1, c2, c3 = st.columns(3)
    c1.metric("M1: PM2.5 only", f"R² = {m1['R2']:.3f}",
              delta=f"RMSE {m1['RMSE']:.2f} yrs", delta_color="off")
    c2.metric("M2: + HDI", f"R² = {m2['R2']:.3f}",
              delta=f"+{m2['R2']-m1['R2']:.3f}", delta_color="normal")
    c3.metric("M3: + interaction", f"R² = {m3['R2']:.3f}",
              delta=f"+{m3['R2']-m2['R2']:.4f}", delta_color="normal")

    st.markdown("### How the PM2.5 coefficient changes")
    st.caption("If HDI is a confounder, the PM2.5 coefficient should "
               "shrink (or vanish) once HDI is added.")

    pm_table = pd.DataFrame({
        "Model": ["M1: PM2.5 only", "M2: + HDI", "M3: + interaction"],
        "PM2.5 coefficient": [t1.loc[1,"coef"], t2.loc[1,"coef"], t3.loc[1,"coef"]],
        "p-value": [t1.loc[1,"p"], t2.loc[1,"p"], t3.loc[1,"p"]],
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pm_table["Model"], y=pm_table["PM2.5 coefficient"],
        text=[f"{v:+.3f}<br>p={p:.3g}"
              for v,p in zip(pm_table["PM2.5 coefficient"], pm_table["p-value"])],
        textposition="outside",
        marker=dict(color=["#ef4444", "#f59e0b", "#10b981"],
                    line=dict(color="white", width=2)),
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
    fig.update_layout(
        height=420, yaxis_title="PM2.5 coefficient (years per µg/m³)",
        title="PM2.5 effect across nested models",
        showlegend=False, margin=dict(l=10,r=10,t=50,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    p_inter = t3.loc[3, "p"]
    if p_inter < 0.05:
        verdict_msg = (f"<b>Moderation is supported</b> in this sample — "
                       f"the PM2.5 × HDI interaction has p = {p_inter:.3f}.")
        cls = "callout-green"
    else:
        verdict_msg = (f"<b>No evidence of moderation</b> in this sample — "
                       f"the PM2.5 × HDI interaction has p = {p_inter:.3f}.")
        cls = "callout-red"
    st.markdown(f'<div class="{cls}">{verdict_msg}</div>',
                unsafe_allow_html=True)

    # Moderation visualisation
    st.markdown("### What does M3 actually predict?")
    st.caption("Predicted PM2.5 → LifeExp slope at three different HDI levels.")

    beta3 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X3]),
                            y, rcond=None)[0]
    hdi_qs = np.quantile(df["HDI"], [0.10, 0.50, 0.90])
    hdi_lbls = [f"Low HDI (10th %, {hdi_qs[0]:.2f})",
                f"Median HDI ({hdi_qs[1]:.2f})",
                f"High HDI (90th %, {hdi_qs[2]:.2f})"]
    pm_grid = np.linspace(df["PM2_5"].min(), df["PM2_5"].max(), 80)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["PM2_5"], y=df["LifeExp"], mode="markers",
                             marker=dict(color="lightgray", size=7,
                                         line=dict(width=0.5,color="white")),
                             name="Countries", hovertext=df["Country"]))
    colors = ["#ef4444", "#f59e0b", "#10b981"]
    for hdi_v, lbl, c in zip(hdi_qs, hdi_lbls, colors):
        pmc  = pm_grid - df["PM2_5"].mean()
        hdic = hdi_v - df["HDI"].mean()
        Xg = np.column_stack([np.ones(80), pmc, np.full(80, hdic), pmc*hdic])
        ypred = Xg @ beta3
        fig.add_trace(go.Scatter(x=pm_grid, y=ypred, mode="lines",
                                 line=dict(color=c, width=3), name=lbl))
    fig.update_layout(height=480, xaxis_title="PM2.5 (µg/m³)",
                      yaxis_title="Life Expectancy (years)",
                      title="M3 predicted slopes at three HDI levels",
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full coefficient tables (M1, M2, M3)"):
        st.markdown("**M1**"); st.dataframe(t1.round(4), use_container_width=True, hide_index=True)
        st.markdown("**M2**"); st.dataframe(t2.round(4), use_container_width=True, hide_index=True)
        st.markdown("**M3** (mean-centered)"); st.dataframe(t3.round(4), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB 5 - MACHINE LEARNING
# -------------------------------------------------------------
with tab5:
    st.markdown("## Random Forest, K-Means and PCA")

    # ---- RANDOM FOREST ----
    st.markdown("###  Random Forest — feature importance")
    st.caption("Non-parametric corroboration of the linear regression. "
               "If HDI dominates here too, the conclusion is robust.")

    X_rf = df[["PM2_5", "HDI"]].values
    y    = df["LifeExp"].values
    rf = RandomForestRegressor(n_estimators=300, random_state=42,
                               min_samples_leaf=2, n_jobs=-1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(rf, X_rf, y, cv=cv, scoring="r2")
    rf.fit(X_rf, y)
    perm = permutation_importance(rf, X_rf, y, n_repeats=20,
                                  random_state=42, n_jobs=-1)

    c1, c2 = st.columns([1, 2])
    c1.metric("CV R²", f"{cv_r2.mean():.3f}",
              delta=f"sd {cv_r2.std():.3f}", delta_color="off")
    c1.metric("HDI / PM2.5 importance ratio",
              f"{perm.importances_mean[1] / max(perm.importances_mean[0], 1e-9):.1f}×")

    fig = go.Figure(go.Bar(
        x=perm.importances_mean,
        y=["PM2.5", "HDI"],
        orientation="h",
        error_x=dict(array=perm.importances_std),
        marker_color=["#3b82f6", "#10b981"],
    ))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10),
                      xaxis_title="Permutation importance (drop in R²)")
    c2.plotly_chart(fig, use_container_width=True)

    # ---- K-MEANS ----
    st.markdown("###  K-Means clustering")
    st.caption("Standardised features: [PM2.5, HDI, LifeExp]. "
               "Elbow + silhouette tell us how many clusters the data "
               "naturally has.")

    feats = ["PM2_5", "HDI", "LifeExp"]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df[feats].values)

    ks = list(range(2, 8))
    inertias = []; silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_std)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_std, labels))
    best_k = ks[int(np.argmax(silhouettes))]

    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                               line=dict(color="#3b82f6", width=3),
                               marker=dict(size=10)))
    fig.update_layout(height=300, title="Elbow (inertia)",
                      xaxis_title="k", yaxis_title="Inertia",
                      margin=dict(l=10,r=10,t=40,b=10))
    c1.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(go.Scatter(x=ks, y=silhouettes, mode="lines+markers",
                               line=dict(color="#f59e0b", width=3),
                               marker=dict(size=10)))
    fig.add_vline(x=best_k, line=dict(color="red", dash="dash"),
                  annotation_text=f"best k = {best_k}",
                  annotation_position="top right")
    fig.update_layout(height=300, title="Silhouette score",
                      xaxis_title="k", yaxis_title="Silhouette",
                      margin=dict(l=10,r=10,t=40,b=10))
    c2.plotly_chart(fig, use_container_width=True)

    # Final clustering with best_k
    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    df_v = df.copy()
    df_v["Cluster"] = km_final.fit_predict(X_std)
    df_v["Cluster"] = df_v["Cluster"].astype(str)

    centers_orig = scaler.inverse_transform(km_final.cluster_centers_)
    centers_df = pd.DataFrame(centers_orig, columns=feats,
                              index=[f"C{i}" for i in range(best_k)])
    centers_df["n"] = df_v["Cluster"].value_counts().sort_index().values
    st.markdown(f"**Cluster profiles (best k = {best_k}):**")
    st.dataframe(centers_df.round(2), use_container_width=True)

    # ---- PCA ----
    st.markdown("###  PCA — 2D summary")
    st.caption("Project the 3 standardised features into 2D so the entire "
               "dataset fits in one chart.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    df_v["PC1"], df_v["PC2"] = X_pca[:,0], X_pca[:,1]

    c1, c2, c3 = st.columns(3)
    c1.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
    c2.metric("PC2 variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
    c3.metric("Total (2 PCs)", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

    color_choice = st.radio("Color the PCA plot by:",
                            ["HDI_Group", "Cluster", "Region"],
                            horizontal=True)
    color_map = HDI_PALETTE if color_choice == "HDI_Group" else None
    cat_order = ({"HDI_Group": HDI_ORDER}
                 if color_choice == "HDI_Group" else None)

    fig = px.scatter(df_v, x="PC1", y="PC2", color=color_choice,
                     color_discrete_map=color_map,
                     category_orders=cat_order,
                     hover_data=["Country","Region","HDI","PM2_5","LifeExp"])
    fig.update_traces(marker=dict(size=11, line=dict(color="white", width=0.6)))
    fig.update_layout(
        height=540,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        margin=dict(l=10,r=10,t=20,b=10),
    )
    # Loading arrows
    scale = 3.5
    for i, feat in enumerate(feats):
        fig.add_annotation(x=pca.components_[0,i]*scale,
                           y=pca.components_[1,i]*scale,
                           ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y",
                           text=feat, showarrow=True, arrowhead=3,
                           arrowwidth=2, arrowcolor="#111827",
                           font=dict(size=12, color="#111827"))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# TAB 6 - COUNTRY EXPLORER
# -------------------------------------------------------------
with tab6:
    st.markdown("## Look up a country")
    st.caption("Pick any country to see its values, where it sits in "
               "the global distribution, and which K-Means cluster it lands in.")

    countries = sorted(df["Country"].tolist())
    if not countries:
        st.info("No countries match the current filters.")
    else:
        selected = st.selectbox("Country:", countries, index=0)
        row = df[df["Country"] == selected].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PM2.5", f"{row['PM2_5']:.1f} µg/m³",
                  delta=f"{row['PM2_5'] - df['PM2_5'].mean():+.1f} vs world avg",
                  delta_color="inverse")
        c2.metric("HDI", f"{row['HDI']:.3f}",
                  delta=f"{row['HDI'] - df['HDI'].mean():+.3f}")
        c3.metric("Life Exp.", f"{row['LifeExp']:.1f} yrs",
                  delta=f"{row['LifeExp'] - df['LifeExp'].mean():+.1f} yrs")
        c4.metric("HDI Group", row["HDI_Group"],
                  delta=row["Region"], delta_color="off")

        # Country position in the PM2.5 vs LifeExp scatter
        fig = px.scatter(df, x="PM2_5", y="LifeExp", color="HDI_Group",
                         color_discrete_map=HDI_PALETTE,
                         category_orders={"HDI_Group": HDI_ORDER},
                         hover_data=["Country"], opacity=0.5)
        fig.update_traces(marker=dict(size=8))
        fig.add_trace(go.Scatter(
            x=[row["PM2_5"]], y=[row["LifeExp"]], mode="markers+text",
            marker=dict(color="black", size=20,
                        line=dict(color="white", width=3)),
            text=[selected], textposition="top center",
            textfont=dict(size=14, color="black"),
            name=selected, showlegend=False
        ))
        fig.update_layout(height=520,
                          title=f"{selected} on the global map",
                          margin=dict(l=10,r=10,t=50,b=10),
                          xaxis_title="PM2.5 (µg/m³)",
                          yaxis_title="Life Expectancy (years)")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.9rem;'>"
    "DSA 210 project · 2020 cross-country analysis · "
    "Data: World Bank, UNDP, Our World in Data"
    "</div>", unsafe_allow_html=True
)
