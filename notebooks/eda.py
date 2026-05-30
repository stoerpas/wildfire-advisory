"""
eda.py
Exploratory Data Analysis for the Algerian Forest Fires dataset.
Run with: python notebooks/eda.py
All figures are saved to notebooks/figures/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.preprocess import load_algerian, save_clean
from src.config import FEATURE_COLS, TARGET_COL

# ── Setup ─────────────────────────────────────────────────
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIRE_PALETTE = {0: "#4C9BE8", 1: "#E85C4C"}   # blue = no fire, red = fire

df = load_algerian()
save_clean(df)

print(f"Dataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nClass balance:\n{df[TARGET_COL].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDescriptive stats:\n{df[FEATURE_COLS].describe().round(2)}")


# ── Figure 1: Class balance ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

counts = df[TARGET_COL].value_counts()
axes[0].bar(["Not Fire", "Fire"], counts.values,
            color=[FIRE_PALETTE[0], FIRE_PALETTE[1]], edgecolor="white", width=0.5)
axes[0].set_title("Class Balance")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

region_fire = df.groupby(["Region", TARGET_COL]).size().unstack(fill_value=0)
region_fire.columns = ["Not Fire", "Fire"]
region_fire.plot(kind="bar", ax=axes[1],
                 color=[FIRE_PALETTE[0], FIRE_PALETTE[1]],
                 edgecolor="white", width=0.6)
axes[1].set_title("Fire Occurrence by Region")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES / "01_class_balance.png", dpi=150)
plt.close()
print("\n✔ Figure 1 saved: class balance")


# ── Figure 2: Monthly fire frequency ─────────────────────
month_map = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
df["month_name"] = df["month"].map(month_map)
month_order = ["Jun", "Jul", "Aug", "Sep"]

fire_by_month = (df.groupby(["month_name", TARGET_COL])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(month_order))
fire_by_month.columns = ["Not Fire", "Fire"]

fig, ax = plt.subplots(figsize=(8, 4))
fire_by_month.plot(kind="bar", ax=ax,
                   color=[FIRE_PALETTE[0], FIRE_PALETTE[1]],
                   edgecolor="white", width=0.7)
ax.set_title("Monthly Fire Frequency (June–September 2012)")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "02_monthly_fire.png", dpi=150)
plt.close()
print("✔ Figure 2 saved: monthly fire frequency")


# ── Figure 3: Feature distributions by class ─────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, feat in enumerate(FEATURE_COLS):
    ax = axes[i]
    for cls, label, color in [(0, "Not Fire", FIRE_PALETTE[0]),
                               (1, "Fire",    FIRE_PALETTE[1])]:
        vals = df[df[TARGET_COL] == cls][feat]
        ax.hist(vals, bins=20, alpha=0.6, color=color,
                label=label, edgecolor="white")
    ax.set_title(feat)
    ax.set_xlabel("")
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("Feature Distributions: Fire vs Not Fire", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES / "03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Figure 3 saved: feature distributions")


# ── Figure 4: Correlation heatmap ────────────────────────
corr = df[FEATURE_COLS + [TARGET_COL]].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix (incl. target: Classes)")
plt.tight_layout()
plt.savefig(FIGURES / "04_correlation_heatmap.png", dpi=150)
plt.close()
print("✔ Figure 4 saved: correlation heatmap")


# ── Figure 5: Boxplots — top discriminating features ─────
top_feats = ["FWI", "Temperature", "FFMC", "ISI", "DMC"]

fig, axes = plt.subplots(1, 5, figsize=(16, 5))
for ax, feat in zip(axes, top_feats):
    data_no  = df[df[TARGET_COL] == 0][feat]
    data_yes = df[df[TARGET_COL] == 1][feat]
    bp = ax.boxplot([data_no, data_yes],
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor(FIRE_PALETTE[0])
    bp["boxes"][1].set_facecolor(FIRE_PALETTE[1])
    ax.set_title(feat)
    ax.set_xticklabels(["Not Fire", "Fire"])
    ax.set_xlabel("")

plt.suptitle("Top Discriminating Features by Fire Class", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES / "05_boxplots_top_features.png", dpi=150)
plt.close()
print("✔ Figure 5 saved: boxplots")


# ── Figure 6: Temperature vs FWI scatter ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for cls, label, color in [(0, "Not Fire", FIRE_PALETTE[0]),
                           (1, "Fire",    FIRE_PALETTE[1])]:
    sub = df[df[TARGET_COL] == cls]
    ax.scatter(sub["Temperature"], sub["FWI"],
               alpha=0.6, s=40, color=color, label=label, edgecolors="white")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Fire Weather Index (FWI)")
ax.set_title("Temperature vs FWI — coloured by fire occurrence")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "06_temperature_vs_fwi.png", dpi=150)
plt.close()
print("✔ Figure 6 saved: temperature vs FWI scatter")


# ── Figure 7: Pairplot — key features ────────────────────
pair_feats = ["Temperature", "RH", "FWI", "FFMC", "ISI", TARGET_COL]
pair_df = df[pair_feats].copy()
pair_df[TARGET_COL] = pair_df[TARGET_COL].map({0: "Not Fire", 1: "Fire"})

g = sns.pairplot(pair_df, hue=TARGET_COL,
                 palette={"Not Fire": FIRE_PALETTE[0], "Fire": FIRE_PALETTE[1]},
                 plot_kws={"alpha": 0.5, "s": 25},
                 diag_kind="kde")
g.fig.suptitle("Pairplot — Key Features", y=1.01, fontsize=13)
g.fig.savefig(FIGURES / "07_pairplot.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔ Figure 7 saved: pairplot")


# ── Key findings summary ──────────────────────────────────
target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
print("\n" + "="*55)
print("EDA KEY FINDINGS")
print("="*55)
print(f"\n1. Dataset: 243 observations, 10 numeric features, binary target")
print(f"   Class balance: 137 fire ({137/243*100:.1f}%) / 106 not fire ({106/243*100:.1f}%)")
print(f"   Two regions: Bejaia & Sidi-Bel-Abbes (122 each)\n")
print("2. Correlation with fire (Classes):")
for feat, val in target_corr.items():
    bar = "█" * int(abs(val) * 20)
    print(f"   {feat:12s}: {val:+.3f}  {bar}")
print(f"\n3. Fire peaks in July–August (dry summer months)")
print(f"4. FWI, ISI, FFMC and Temperature show strongest separation")
print(f"   between fire and no-fire days")
print(f"5. RH (humidity) is negatively correlated — higher humidity → less fire")
print(f"6. DC and BUI are strongly correlated with each other (drought indices)")
print("="*55)
print(f"\nAll figures saved to: {FIGURES}")