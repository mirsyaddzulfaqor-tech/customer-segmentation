"""Builds and executes customer_segmentation.ipynb (run: python build_notebook.py)."""
import sys
sys.path.insert(0, ".")
from nbrun import Notebook

nb = Notebook(".")

nb.md("""
# Customer Segmentation with K-Means

**Business question:** our customers are not all the same. Which groups exist, and how should marketing talk to each one?

This notebook rebuilds my Digital Skola *Conqueror Project 2*: clean the data, choose the number of clusters,
segment 2,000 customers with K-Means, and turn the clusters into personas with marketing actions.

**Data:** 2,000 customers, 7 attributes: sex, marital status, age, education, income, occupation and settlement size
(`data/customer_segmentation.csv`, a public teaching dataset).
""")

nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
""")

nb.md("## 1. Load and check the data")
nb.code("""
raw = pd.read_csv("data/customer_segmentation.csv")
print(raw.shape, "| missing values:", raw.isna().sum().sum(), "| duplicates:", raw.duplicated().sum())
raw.head()
""")
nb.code("""
# Codes follow the dataset legend.
LEGEND = {
    "Sex": {0: "Male", 1: "Female"},
    "Marital status": {0: "Single", 1: "Non-single"},
    "Education": {0: "Other/unknown", 1: "High school", 2: "University", 3: "Graduate"},
    "Occupation": {0: "Unemployed/unskilled", 1: "Skilled employee", 2: "Management/self-employed"},
    "Settlement size": {0: "Small city", 1: "Mid-sized city", 2: "Big city"},
}
features = raw.drop(columns="ID")
labels = features.copy()
for col, mapping in LEGEND.items():
    labels[col] = labels[col].map(mapping)
features.describe().round(1)
""")

nb.md("## 2. Explore the customers")
nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
sns.histplot(features["Age"], bins=30, ax=axes[0], color="#4C72B0")
axes[0].set_title("Age distribution")
sns.histplot(features["Income"], bins=30, ax=axes[1], color="#4C72B0")
axes[1].set_title("Annual income distribution")
plt.tight_layout()
print("Correlation age vs income:", round(features["Age"].corr(features["Income"]), 2))
""")
nb.code("""
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, col in zip(axes, LEGEND):
    labels[col].value_counts().plot(kind="bar", ax=ax, color="#4C72B0", rot=30)
    ax.set_title(col)
plt.tight_layout()
""")
nb.md("""
Most customers are in their late 20s to 40s, earn around 120,000 a year, finished high school and work as skilled employees.
Age and income are moderately correlated (0.34). This is not enough to separate groups by eye, so clustering is used.
""")

nb.md("""
## 3. How many segments?

All seven attributes are standardised so that income (in the hundreds of thousands) does not dominate the
coded variables. Two checks are used: the **elbow** of the within-cluster error, and the **silhouette score**.
""")
nb.code("""
X = StandardScaler().fit_transform(features)
ks = range(2, 9)
inertia, silhouette = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(list(ks), inertia, marker="o"); axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
axes[1].plot(list(ks), silhouette, marker="o", color="#55A868"); axes[1].set(title="Silhouette score", xlabel="k")
axes[0].axvline(4, color="#C44E52", ls="--"); axes[1].axvline(4, color="#C44E52", ls="--")
plt.tight_layout()
pd.DataFrame({"k": list(ks), "inertia": np.round(inertia), "silhouette": np.round(silhouette, 3)}).set_index("k")
""")
nb.md("""
The elbow bends around **k = 4**. Silhouette scores are close between 3 and 7 (0.25–0.29), so the data has no
single "natural" k. Four segments were chosen because they are distinct enough to describe and few enough for
a marketing team to run separate campaigns.
""")

nb.md("## 4. Build the segments")
nb.code("""
kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(X)
labels["Cluster"] = kmeans.labels_
features["Cluster"] = kmeans.labels_

profile = features.groupby("Cluster").agg(
    customers=("Age", "size"),
    avg_age=("Age", "mean"),
    avg_income=("Income", "mean"),
    share_female=("Sex", "mean"),
    share_non_single=("Marital status", "mean"),
)
profile["share_of_base"] = profile["customers"] / profile["customers"].sum()
profile.round(2)
""")
nb.code("""
def mix(col):
    return pd.crosstab(labels["Cluster"], labels[col], normalize="index").mul(100).round(0)

for col in ["Education", "Occupation", "Settlement size"]:
    print(f"\\n{col} (% of each cluster)")
    print(mix(col).to_string())
""")

nb.code("""
# Name the clusters from their profiles (checked against the tables above).
PERSONAS = {
    0: "Young family women",
    1: "Urban career men",
    2: "Budget-conscious small-town singles",
    3: "Affluent mature professionals",
}
assert profile.loc[3, "avg_age"] > 50 and profile.loc[0, "share_non_single"] > 0.9
labels["Persona"] = labels["Cluster"].map(PERSONAS)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pts = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=labels["Persona"], s=18, alpha=0.7, palette="Set2")
plt.title(f"Customer segments (PCA view, {pca.explained_variance_ratio_.sum():.0%} of variance shown)")
plt.xlabel("Component 1"); plt.ylabel("Component 2")
plt.legend(title="", fontsize=8, loc="best")
""")
nb.code("""
plt.figure(figsize=(8, 5))
sns.scatterplot(data=labels, x="Age", y="Income", hue="Persona", s=18, alpha=0.7, palette="Set2")
plt.title("Segments by age and income")
plt.legend(title="", fontsize=8)
""")

nb.md("""
## 5. Personas and marketing actions

| Segment | Share | Who they are | Suggested approach |
| --- | --- | --- | --- |
| **Young family women** | 35% | Around 29, almost all non-single, 85% women, high-school educated, mostly in small cities, mid income (~106k) | Family bundles, instalment plans, social-media and community channels |
| **Urban career men** | 28% | Around 36, 97% men, mostly single, skilled or management roles, mid/big cities, high income (~141k) | Premium and convenience products, time-saving services, digital and app-first channels |
| **Budget-conscious small-town singles** | 23% | Around 36, single, 96% in small cities, two-thirds unemployed or unskilled, lowest income (~98k) | Entry-level and value products, promotions and discounts, local partners |
| **Affluent mature professionals** | 13% | Around 56, university or graduate degree, highest income (~158k), spread across city sizes | High-value products, loyalty and personal service, financial or health-related offers |

**How to use it:** tag each customer with their segment in the CRM, track sales and response by segment, and test one tailored campaign per segment against a generic one.

*Limitations:* K-Means needs a chosen k, and silhouette scores (~0.27) show the segments overlap. The personas are a practical simplification, not hard boundaries.
""")

nb.code("""
summary = labels.groupby("Persona").agg(customers=("Age", "size"), avg_age=("Age", "mean"),
                                       avg_income=("Income", "mean")).round(0)
summary["share"] = (summary["customers"] / summary["customers"].sum()).round(2)
summary.sort_values("customers", ascending=False)
""")

nb.save("customer_segmentation.ipynb")
print("saved")
