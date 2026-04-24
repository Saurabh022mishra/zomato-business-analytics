"""
=============================================================================
BE277 – Business Analytics for Managers and Entrepreneurs
Zomato Restaurants Dataset – Full Analysis Script
University of Essex, Essex Business School
Academic Year: 2026, Spring Term
=============================================================================

REQUIREMENTS
------------
Install dependencies before running:
    pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

FILES NEEDED (place in same directory as this script):
    - zomato.csv
    - Country_Code.xlsx

OUTPUT
------
Running this script will:
    1. Print summary statistics and model metrics to the console
    2. Save all 9 figures to: ./outputs/figures/
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 – IMPORTS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
matplotlib.use('Agg')   # Use non-interactive backend (safe for all environments)

# ── Colour palette (consistent across all figures) ───────────────────────────
C1   = '#003f7f'   # deep navy
C2   = '#c0392b'   # crimson
C3   = '#2980b9'   # mid-blue
C4   = '#27ae60'   # green
C5   = '#8e44ad'   # purple
C6   = '#e67e22'   # orange
GREY = '#95a5a6'
PALETTE = [C1, C2, C3, C4, C5, C6, '#1abc9c', '#e74c3c']

# ── Matplotlib global style ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    '#fafafa',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    '#cccccc',
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.alpha':        0.4,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'legend.framealpha': 0.8,
})

# ── Output directory ──────────────────────────────────────────────────────────
OUT = './outputs/figures'
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# SECTION 1 – DATA LOADING AND CLEANING
# =============================================================================

def load_data(zomato_path='data/zomato.csv', country_path='data/Country_Code.xlsx'):
    """
    Load and merge the Zomato and Country Code datasets.

    Steps:
        1. Read zomato.csv with latin-1 encoding (handles special characters).
        2. Read Country_Code.xlsx for country name lookup.
        3. Merge on 'Country Code' to add a human-readable 'Country' column.
        4. Drop the 9 rows with missing Cuisines (< 0.1% of data).
        5. Encode binary service flags as integers for modelling.

    Returns
    -------
    df        : pd.DataFrame – full cleaned dataset (9,542 rows)
    df_rated  : pd.DataFrame – subset excluding 'Not rated' rows (7,394 rows)
    """
    df = pd.read_csv('data/zomato.csv', encoding='latin-1')
    cc = pd.read_excel(country_path)

    # Merge country names
    df = df.merge(cc, on='Country Code', how='left')

    # Remove the small number of missing-cuisine rows
    df = df.dropna(subset=['Cuisines']).reset_index(drop=True)

    # Binary encode service flags
    df['Online_Delivery'] = (df['Has Online delivery'] == 'Yes').astype(int)
    df['Table_Booking']   = (df['Has Table booking']   == 'Yes').astype(int)

    # Subset of rated restaurants only (Aggregate rating > 0)
    df_rated = df[df['Aggregate rating'] > 0].copy()

    return df, df_rated


print("Loading data...")
df, df_rated = load_data()

print(f"  Total records (after cleaning): {len(df):,}")
print(f"  Rated records (rating > 0):     {len(df_rated):,}")
print(f"  Countries:                       {df['Country'].nunique()}")
print(f"  Cities:                          {df['City'].nunique()}")
print(f"  Cuisine types:                   {len(set(c.strip() for row in df['Cuisines'] for c in row.split(','))):,}")


# =============================================================================
# SECTION 2 – DESCRIPTIVE ANALYTICS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Geographic Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_geography(df):
    """
    Figure 1: Side-by-side bar chart (restaurant count by country) and
    pie chart (market share, grouping countries 7–15 as 'Other').
    """
    country_counts = df['Country'].value_counts()
    top_other = country_counts[:6].copy()
    top_other['Other'] = country_counts[6:].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 1: Geographic Distribution of Restaurants',
                 fontsize=13, fontweight='bold', y=1.01)

    # Bar chart (all 15 countries)
    bar_colors = [C1 if c == 'India' else C3 for c in country_counts.index[::-1]]
    axes[0].barh(country_counts.index[::-1], country_counts.values[::-1],
                 color=bar_colors, edgecolor='white', height=0.7)
    axes[0].set_title('(a) Count by Country')
    axes[0].set_xlabel('Number of Restaurants')
    for i, (idx, v) in enumerate(zip(country_counts.index[::-1],
                                      country_counts.values[::-1])):
        axes[0].text(v + 20, i, f'{v:,}', va='center', fontsize=8)

    # Pie chart
    wedge_colors = [C1] + [C3] * 4 + [C4, GREY]
    axes[1].pie(top_other.values, labels=top_other.index,
                colors=wedge_colors, autopct='%1.1f%%', startangle=140,
                wedgeprops=dict(edgecolor='white', linewidth=1.2),
                textprops={'fontsize': 8.5})
    axes[1].set_title('(b) Market Share')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_geography.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_geography.png")


def plot_top_cities(df):
    """
    Figure 2: Horizontal bar chart of the top 10 cities by restaurant count.
    Colour-codes rank 1 (crimson), 2–3 (navy), 4–10 (blue).
    """
    top_cities = df['City'].value_counts().head(10)
    colors_c = [C2 if i == 0 else C1 if i < 3 else C3
                for i in range(len(top_cities))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_cities.index, top_cities.values,
                  color=colors_c, edgecolor='white', width=0.7)
    ax.set_title('Figure 2: Top 10 Cities by Number of Listed Restaurants')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Restaurants')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f'{int(bar.get_height()):,}',
                ha='center', fontsize=8, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    patches = [mpatches.Patch(color=C2, label='Rank #1'),
               mpatches.Patch(color=C1, label='Rank #2–3'),
               mpatches.Patch(color=C3, label='Rank #4–10')]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_cities.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_cities.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Ratings and Engagement
# ─────────────────────────────────────────────────────────────────────────────

def plot_ratings(df, df_rated):
    """
    Figure 3: Three-panel ratings analysis:
        (a) Histogram with mean/median lines
        (b) Bar chart of rating text categories
        (c) Votes distribution (capped at 2,000 for readability)
    """
    rating_order = ['Poor', 'Average', 'Good', 'Very Good', 'Excellent', 'Not rated']
    rc = df['Rating text'].value_counts().reindex(rating_order).fillna(0)
    bar_colors = [C2, C6, GREY, C3, C4, '#bdc3c7']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 3: Ratings and Engagement Analysis',
                 fontsize=13, fontweight='bold')

    # (a) Histogram
    axes[0].hist(df_rated['Aggregate rating'], bins=24,
                 color=C3, edgecolor='white', alpha=0.85)
    axes[0].axvline(df_rated['Aggregate rating'].mean(), color=C2,
                    linestyle='--', linewidth=2,
                    label=f"Mean = {df_rated['Aggregate rating'].mean():.2f}")
    axes[0].axvline(df_rated['Aggregate rating'].median(), color=C4,
                    linestyle=':', linewidth=2,
                    label=f"Median = {df_rated['Aggregate rating'].median():.2f}")
    axes[0].set_title('(a) Rating Distribution')
    axes[0].set_xlabel('Aggregate Rating')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # (b) Category bar
    axes[1].bar(rc.index, rc.values, color=bar_colors,
                edgecolor='white', width=0.7)
    for i, v in enumerate(rc.values):
        axes[1].text(i, v + 30, f'{int(v):,}', ha='center', fontsize=8)
    axes[1].set_title('(b) Rating Category Counts')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Count')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25, ha='right')

    # (c) Votes histogram (capped)
    axes[2].hist(df_rated['Votes'].clip(upper=2000), bins=40,
                 color=C5, edgecolor='white', alpha=0.85)
    median_v = df_rated['Votes'].median()
    axes[2].axvline(median_v, color=C2, linestyle='--', linewidth=2,
                    label=f'Median = {median_v:.0f}')
    axes[2].set_title('(c) Votes Distribution (capped at 2,000)')
    axes[2].set_xlabel('Votes')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_ratings.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_ratings.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Cuisine and Menu Insights
# ─────────────────────────────────────────────────────────────────────────────

def plot_cuisines(df):
    """
    Figure 4: Horizontal bar chart of the 15 most frequently listed cuisines.
    Cuisines are extracted from comma-separated multi-label strings.
    Colour highlights the top 3 (crimson), 4–7 (navy), and 8–15 (blue).
    """
    cuisine_list = []
    for entry in df['Cuisines'].dropna():
        cuisine_list.extend([c.strip() for c in entry.split(',')])
    counts = Counter(cuisine_list)
    top_c = pd.Series(dict(counts.most_common(15)))

    colors_c = [C2 if i < 3 else C1 if i < 7 else C3
                for i in range(len(top_c))]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top_c.index[::-1], top_c.values[::-1],
            color=colors_c[::-1], edgecolor='white', height=0.7)
    ax.set_title('Figure 4: Top 15 Most Listed Cuisines on Zomato')
    ax.set_xlabel('Number of Restaurant Listings')
    for i, (idx, v) in enumerate(zip(top_c.index[::-1], top_c.values[::-1])):
        ax.text(v + 15, i, f'{int(v):,}', va='center', fontsize=8.5)
    patches = [mpatches.Patch(color=C2, label='Top 3'),
               mpatches.Patch(color=C1, label='Rank 4–7'),
               mpatches.Patch(color=C3, label='Rank 8–15')]
    ax.legend(handles=patches, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_cuisines.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_cuisines.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Pricing and Value
# ─────────────────────────────────────────────────────────────────────────────

def plot_pricing(df, df_rated):
    """
    Figure 5: Three-panel pricing analysis:
        (a) Restaurant count per price tier
        (b) Box plot of ratings by tier
        (c) Mean rating ± standard deviation per tier
    """
    price_labels = {1: 'Budget (1)', 2: 'Mid-range (2)',
                    3: 'Premium (3)', 4: 'Fine Dining (4)'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 5: Pricing, Value, and Rating Relationship',
                 fontsize=13, fontweight='bold')

    # (a) Count
    pr_counts = df['Price range'].value_counts().sort_index()
    axes[0].bar([price_labels[i] for i in pr_counts.index],
                pr_counts.values, color=PALETTE[:4],
                edgecolor='white', width=0.6)
    for i, v in enumerate(pr_counts.values):
        axes[0].text(i, v + 40, f'{v:,}', ha='center',
                     fontsize=9, fontweight='bold')
    axes[0].set_title('(a) Count by Price Tier')
    axes[0].set_ylabel('Number of Restaurants')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15)

    # (b) Box plot
    groups = [df_rated[df_rated['Price range'] == i]['Aggregate rating']
              for i in [1, 2, 3, 4]]
    bp = axes[1].boxplot(groups, patch_artist=True, notch=False,
                          labels=['Budget', 'Mid', 'Premium', 'Fine'],
                          medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[1].set_title('(b) Rating Distribution by Price Tier')
    axes[1].set_xlabel('Price Tier')
    axes[1].set_ylabel('Aggregate Rating')

    # (c) Mean ± std
    mean_r = df_rated.groupby('Price range')['Aggregate rating'].mean()
    std_r  = df_rated.groupby('Price range')['Aggregate rating'].std()
    axes[2].bar([price_labels[i] for i in mean_r.index],
                mean_r.values, yerr=std_r.values,
                color=PALETTE[:4], edgecolor='white', capsize=5, width=0.6)
    for i, v in enumerate(mean_r.values):
        axes[2].text(i, v + 0.05, f'{v:.2f}', ha='center',
                     fontsize=10, fontweight='bold')
    axes[2].set_title('(c) Mean Rating ± Std Dev by Price Tier')
    axes[2].set_ylabel('Mean Rating')
    axes[2].set_ylim(0, 5.2)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_pricing.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_pricing.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Online Delivery and Services
# ─────────────────────────────────────────────────────────────────────────────

def plot_online_delivery(df, df_rated):
    """
    Figure 6: Three-panel online delivery analysis:
        (a) Pie chart – proportion with/without online delivery
        (b) Bar chart – mean rating for delivery vs no-delivery
        (c) Grouped bar – delivery impact across top-5 countries
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 6: Online Delivery and Services Analysis',
                 fontsize=13, fontweight='bold')

    # (a) Pie
    del_counts = df['Has Online delivery'].value_counts()
    axes[0].pie(del_counts.values,
                labels=['No Delivery', 'Has Delivery'],
                colors=[GREY, C2], autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1.5))
    axes[0].set_title('(a) Online Delivery Availability')

    # (b) Mean rating
    del_avg = df_rated.groupby('Has Online delivery')['Aggregate rating'] \
                      .agg(['mean', 'std'])
    bars = axes[1].bar(['No Delivery', 'Has Delivery'],
                       del_avg['mean'].values,
                       yerr=del_avg['std'].values,
                       color=[GREY, C2], edgecolor='white',
                       capsize=6, width=0.5)
    for bar, v in zip(bars, del_avg['mean'].values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04,
                     f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Average Rating: Delivery vs No Delivery')
    axes[1].set_ylabel('Average Aggregate Rating')
    axes[1].set_ylim(0, 5)

    # (c) By country
    top5 = df['Country'].value_counts().head(5).index
    sub = df_rated[df_rated['Country'].isin(top5)]
    pivot = sub.groupby(['Country', 'Has Online delivery'])['Aggregate rating'] \
               .mean().unstack()
    pivot.columns = ['No Delivery', 'Has Delivery']
    x = np.arange(len(pivot))
    w = 0.35
    axes[2].bar(x - w / 2, pivot['No Delivery'], w,
                label='No Delivery', color=GREY, edgecolor='white')
    axes[2].bar(x + w / 2, pivot['Has Delivery'], w,
                label='Has Delivery', color=C2, edgecolor='white')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pivot.index, rotation=20, ha='right')
    axes[2].set_title('(c) Delivery Impact by Top 5 Countries')
    axes[2].set_ylabel('Average Rating')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_delivery.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_delivery.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation(df_rated):
    """
    Figure 7: Lower-triangle Pearson correlation heatmap using seaborn.
    Covers: rating, votes, average cost, price range, online delivery,
    and table booking.
    """
    corr_cols = ['Aggregate rating', 'Votes', 'Average Cost for two',
                 'Price range', 'Online_Delivery', 'Table_Booking']
    labels = ['Rating', 'Votes', 'Cost/2',
              'Price Range', 'Online Del.', 'Table Book.']
    corr_mat = df_rated[corr_cols].corr()

    # Mask upper triangle (keep lower + diagonal)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.8, ax=ax, mask=mask,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title('Figure 7: Pearson Correlation Matrix — Key Variables',
                 pad=14, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_heatmap.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig7_heatmap.png")


# Run all descriptive plots
print("\n[DESCRIPTIVE ANALYTICS] Generating figures...")
plot_geography(df)
plot_top_cities(df)
plot_ratings(df, df_rated)
plot_cuisines(df)
plot_pricing(df, df_rated)
plot_online_delivery(df, df_rated)
plot_correlation(df_rated)


# =============================================================================
# SECTION 3 – PREDICTIVE ANALYTICS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Data Preparation for Modelling
# ─────────────────────────────────────────────────────────────────────────────

print("\n[PREDICTIVE ANALYTICS] Preparing model data...")

# Encode Country as integer label
le = LabelEncoder()
df_rated['Country_enc'] = le.fit_transform(df_rated['Country'])

# Feature set and target
FEATURES = ['Price range', 'Average Cost for two', 'Votes',
            'Online_Delivery', 'Table_Booking', 'Country_enc']
TARGET   = 'Aggregate rating'

X = df_rated[FEATURES]
y = df_rated[TARGET]

# 80 / 20 train-test split (stratified by rating text for balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training set: {len(X_train):,} rows")
print(f"  Test set:     {len(X_test):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Model Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, cv=5):
    """
    Train a model, predict on the test set, and compute R², MAE, RMSE.
    Also runs k-fold cross-validation on the full dataset.

    Parameters
    ----------
    name   : str          – display name
    model  : estimator    – sklearn model instance
    cv     : int          – number of cross-validation folds

    Returns
    -------
    dict with keys: R2, MAE, RMSE, CV_mean, CV_std, y_pred
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"  {name:<22} R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse,
            'CV_mean': cv_scores.mean(), 'CV_std': cv_scores.std(),
            'y_pred': y_pred, 'model': model}


print("\n  Training models (this may take ~30 seconds)...")
results = {
    'Linear Regression':  evaluate_model(
        'Linear Regression',
        LinearRegression(),
        X_train, y_train, X_test, y_test
    ),
    'Random Forest':      evaluate_model(
        'Random Forest',
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        X_train, y_train, X_test, y_test
    ),
    'Gradient Boosting':  evaluate_model(
        'Gradient Boosting',
        GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42),
        X_train, y_train, X_test, y_test
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Model Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_evaluation(results, X_test, y_test, feature_names):
    """
    Figure 8: Five-panel model evaluation dashboard:
        (a) R² comparison bar chart
        (b) MAE and RMSE comparison (grouped bars)
        (c) Cross-validation R² box plot (Random Forest)
        (d) Feature importance (Random Forest)
        (e) Actual vs Predicted scatter (Gradient Boosting – best model)
    """
    names = list(results.keys())
    r2s   = [results[n]['R2']   for n in names]
    maes  = [results[n]['MAE']  for n in names]
    rmses = [results[n]['RMSE'] for n in names]

    rf_model = results['Random Forest']['model']
    gb_model = results['Gradient Boosting']['model']

    # Re-run CV to get individual fold scores for the box plot
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle('Figure 8: Predictive Model Evaluation Dashboard',
                 fontsize=14, fontweight='bold', y=1.01)

    # (a) R² bar
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(names, r2s, color=[GREY, C3, C2],
                   edgecolor='white', width=0.55)
    for bar, v in zip(bars, r2s):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax0.set_title('(a) Model Comparison — R²')
    ax0.set_ylabel('R² Score')
    ax0.set_ylim(0, 0.75)
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=12, ha='right', fontsize=8.5)

    # (b) MAE & RMSE
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w / 2, maes,  w, label='MAE',  color=C3, edgecolor='white')
    ax1.bar(x + w / 2, rmses, w, label='RMSE', color=C5, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=12, ha='right', fontsize=8.5)
    ax1.set_title('(b) Error Metrics by Model')
    ax1.set_ylabel('Error (rating scale 0–5)')
    ax1.legend()

    # (c) CV box plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.boxplot(cv_scores, patch_artist=True,
                boxprops=dict(facecolor=C3, alpha=0.6),
                medianprops=dict(color=C2, linewidth=2))
    for score in cv_scores:
        ax2.scatter(1, score, color=C2, zorder=5, s=40, alpha=0.8)
    ax2.set_title('(c) Random Forest — 5-Fold CV R²')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels(['RF Cross-Val'])

    # (d) Feature importance
    ax3 = fig.add_subplot(gs[1, 0:2])
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)
    imp_colors = [C2 if imp[i] == max(imp) else C1 if imp[i] > np.mean(imp)
                  else C3 for i in idx]
    ax3.barh([feature_names[i] for i in idx], imp[idx],
             color=imp_colors, edgecolor='white', height=0.6)
    ax3.set_title('(d) Feature Importance — Random Forest')
    ax3.set_xlabel('Importance Score')
    for i, v in enumerate(imp[idx]):
        ax3.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # (e) Actual vs Predicted (Gradient Boosting)
    ax4 = fig.add_subplot(gs[1, 2])
    yp_gb = gb_model.predict(X_test)
    ax4.scatter(y_test, yp_gb, alpha=0.25, color=C2, s=12, label='Predictions')
    lims = [min(y_test.min(), yp_gb.min()),
            max(y_test.max(), yp_gb.max())]
    ax4.plot(lims, lims, color=C1, linewidth=2, label='Perfect Fit')
    ax4.set_title('(e) Actual vs Predicted — Gradient Boosting')
    ax4.set_xlabel('Actual Rating')
    ax4.set_ylabel('Predicted Rating')
    r2_gb = results['Gradient Boosting']['R2']
    mae_gb = results['Gradient Boosting']['MAE']
    ax4.text(0.05, 0.90,
             f'R² = {r2_gb:.3f}\nMAE = {mae_gb:.3f}',
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
             fontsize=9)
    ax4.legend(fontsize=8)

    plt.savefig(f'{OUT}/fig8_model_evaluation.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8_model_evaluation.png")


feat_labels = ['Price Range', 'Avg Cost/2', 'Votes',
               'Online Delivery', 'Table Booking', 'Country']
print("\n[PREDICTIVE ANALYTICS] Generating model evaluation figure...")
plot_model_evaluation(results, X_test, y_test, feat_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(df_rated, k=4):
    """
    Segment restaurants into k clusters using K-Means on four features:
        Price Range, Aggregate Rating, Votes, Online Delivery.

    Steps:
        1. Standardise features with z-score normalisation.
        2. Run elbow method (k = 2–8) to select optimal k.
        3. Compute silhouette scores to validate separation.
        4. Fit final model with selected k.
        5. Reduce to 2D with PCA for visualisation.
        6. Compute and print cluster profiles.

    Parameters
    ----------
    k : int – number of clusters (default 4 based on elbow analysis)

    Returns
    -------
    Xc      : pd.DataFrame with 'Cluster' column added
    Xpca    : np.ndarray – 2D PCA projection
    cp      : pd.DataFrame – mean feature values per cluster
    exp_var : list – PCA explained variance ratios
    """
    cluster_features = ['Price range', 'Aggregate rating', 'Votes', 'Online_Delivery']
    Xc = df_rated[cluster_features].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)

    # Elbow and silhouette
    inertias, sil_scores = [], []
    for ki in range(2, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(Xs, lbl))
    print(f"\n  Silhouette scores by k: "
          f"{dict(zip(range(2, 9), [round(s, 3) for s in sil_scores]))}")

    # Final fit
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_final.fit_predict(Xs)
    Xc['Cluster'] = labels

    # PCA projection
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xs)
    exp_var = pca.explained_variance_ratio_

    # Cluster profiles
    cp = Xc.groupby('Cluster')[cluster_features].mean().round(2)
    print("\n  Cluster Profiles:")
    print(cp.to_string())

    return Xc, Xpca, labels, cp, exp_var, inertias, sil_scores


def plot_clustering(Xc, Xpca, labels, cp, exp_var, inertias, sil_scores, k=4):
    """
    Figure 9: Four-panel clustering visualisation:
        (a) Elbow method – inertia vs k
        (b) Silhouette score vs k
        (c) PCA 2D scatter of clusters
        (d) Normalised cluster profile bar chart
    """
    cluster_palette = [C1, C2, C3, C4]
    feat_names = ['Price Range', 'Avg Rating', 'Votes', 'Online Del.']

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Figure 9: K-Means Clustering Analysis (k={k})',
                 fontsize=14, fontweight='bold')

    # (a) Elbow
    axes[0, 0].plot(range(2, 9), inertias, 'o-', color=C1,
                    linewidth=2, markersize=8)
    axes[0, 0].axvline(k, color=C2, linestyle='--', linewidth=1.5,
                       label=f'Selected k={k}')
    axes[0, 0].set_title('(a) Elbow Method — Inertia vs k')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia (SSE)')
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(2, 9))

    # (b) Silhouette
    axes[0, 1].plot(range(2, 9), sil_scores, 's-', color=C4,
                    linewidth=2, markersize=8)
    axes[0, 1].axvline(k, color=C2, linestyle='--', linewidth=1.5,
                       label=f'Selected k={k}')
    axes[0, 1].set_title('(b) Silhouette Score vs k')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(2, 9))

    # (c) PCA scatter
    for cl in range(k):
        mask = labels == cl
        axes[1, 0].scatter(Xpca[mask, 0], Xpca[mask, 1],
                            c=cluster_palette[cl], alpha=0.4,
                            s=15, label=f'Cluster {cl}')
    axes[1, 0].set_title(
        f'(c) Clusters in PCA Space\n'
        f'(PC1={exp_var[0]*100:.1f}%, PC2={exp_var[1]*100:.1f}%)'
    )
    axes[1, 0].set_xlabel('Principal Component 1')
    axes[1, 0].set_ylabel('Principal Component 2')
    axes[1, 0].legend(markerscale=2, fontsize=9)

    # (d) Normalised profiles
    cp_num = cp.values
    cp_norm = (cp_num - cp_num.min(axis=0)) / \
              (cp_num.max(axis=0) - cp_num.min(axis=0) + 1e-9)
    x = np.arange(len(feat_names))
    w = 0.2
    for i in range(k):
        axes[1, 1].bar(x + i * w, cp_norm[i], w,
                       label=f'Cluster {i}',
                       color=cluster_palette[i],
                       edgecolor='white', alpha=0.85)
    axes[1, 1].set_xticks(x + w * 1.5)
    axes[1, 1].set_xticklabels(feat_names, rotation=15, ha='right')
    axes[1, 1].set_title('(d) Normalised Cluster Profiles')
    axes[1, 1].set_ylabel('Normalised Score (0–1)')
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig9_clustering.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved: fig9_clustering.png")


print("\n[CLUSTERING] Running K-Means analysis...")
Xc, Xpca, labels, cp, exp_var, inertias, sil_scores = run_clustering(df_rated, k=4)
print("\n[CLUSTERING] Generating clustering figure...")
plot_clustering(Xc, Xpca, labels, cp, exp_var, inertias, sil_scores, k=4)


# =============================================================================
# SECTION 4 – SUMMARY OUTPUT
# =============================================================================

print("\n" + "="*65)
print("ANALYSIS COMPLETE — Summary of Key Findings")
print("="*65)
print(f"\n  Dataset:      {len(df):,} restaurants | {df['Country'].nunique()} countries")
print(f"  Rated subset: {len(df_rated):,} restaurants with ratings > 0")

print("\n  Model Performance:")
for name, res in results.items():
    print(f"    {name:<22}  R²={res['R2']:.4f}  MAE={res['MAE']:.4f}")

print(f"\n  Best model: Gradient Boosting (R²={results['Gradient Boosting']['R2']:.4f})")

print("\n  Cluster Archetypes (k=4):")
archetype_names = {
    0: 'Budget, Low-Engagement',
    1: 'Budget, Delivery-Enabled',
    2: 'Premium, High-Engagement',
    3: 'Premium, Niche/Specialist'
}
for cl in range(4):
    print(f"    Cluster {cl} — {archetype_names[cl]}  "
          f"(Rating={cp.loc[cl,'Aggregate rating']:.2f}, "
          f"Votes={cp.loc[cl,'Votes']:.0f})")

print(f"\n  All figures saved to: {OUT}/")
print("="*65)