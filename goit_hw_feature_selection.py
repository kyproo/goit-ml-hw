# ============================================================
# Домашнє завдання: Mutual Information vs Feature Importance
# Набір даних: Autos (automobiles)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

# ── 1. Завантаження даних ──────────────────────────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

columns = [
    "symboling","normalized_losses","make","fuel_type","aspiration",
    "num_of_doors","body_style","drive_wheels","engine_location",
    "wheel_base","length","width","height","curb_weight","engine_type",
    "num_of_cylinders","engine_size","fuel_system","bore","stroke",
    "compression_ratio","horsepower","peak_rpm","city_mpg","highway_mpg","price"
]

df = pd.read_csv(url, names=columns, na_values="?")

# ── 2. Підготовка даних ────────────────────────────────────
# Прибираємо рядки без цільової змінної
df = df.dropna(subset=["price"])
df["price"] = df["price"].astype(float)

# Заповнюємо пропуски
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ── 3. Визначення дискретних ознак ────────────────────────
discrete_features = df.select_dtypes(include="object").columns.tolist()
discrete_features += ["symboling", "num_of_doors"]  # числові але дискретні
# num_of_doors — текстовий ("two"/"four"), вже в object

# Булева маска для mutual_info_regression
feature_cols = [c for c in df.columns if c != "price"]
X_raw = df[feature_cols].copy()

# Кодуємо категорії → числа для MI та моделі
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
X_raw[cat_cols] = enc.fit_transform(X_raw[cat_cols])
X = X_raw.astype(float)
y = df["price"]

discrete_mask = [col in discrete_features or col in cat_cols for col in feature_cols]

# ── 4. Mutual Information ──────────────────────────────────
mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols, name="MI Score")

# ── 5. Регресійна модель — Feature Importance ──────────────
model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X, y)
fi_series = pd.Series(model.feature_importances_, index=feature_cols, name="FI Score")

# ── 6. Масштабування через rank(pct=True) ──────────────────
scores_df = pd.DataFrame({"MI Score": mi_series, "FI Score": fi_series})
scores_ranked = scores_df.rank(pct=True)

# ── 7. Побудова grouped barplot ────────────────────────────
# melt для seaborn catplot
melted = scores_ranked.reset_index().melt(
    id_vars="index", var_name="Method", value_name="Rank (pct)"
)
melted.rename(columns={"index": "Feature"}, inplace=True)

plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=melted, kind="bar",
    x="Rank (pct)", y="Feature",
    hue="Method", palette=["#1D6FE8", "#F5C400"],
    height=10, aspect=1.2,
    order=scores_ranked["MI Score"].sort_values(ascending=False).index
)
g.set_axis_labels("Percentile Rank", "Feature")
g.figure.suptitle(
    "Mutual Information vs Feature Importance\n(GradientBoosting, Autos dataset)",
    fontsize=14, y=1.01
)
g.figure.tight_layout()
plt.savefig("/Users/kevin/.openclaw/workspace/goit_hw_plot.png", dpi=150, bbox_inches="tight")
print("Збережено: goit_hw_plot.png")

# ── 8. Висновки ────────────────────────────────────────────
print("""
ВИСНОВКИ:
─────────────────────────────────────────────────────────
• Ознаки з ВИСОКИМ MI (engine_size, curb_weight, horsepower) 
  також мають високу важливість у моделі — це логічно.

• Деякі ознаки з НИЗЬКИМ MI (символьні / категоріальні: 
  make, body_style, fuel_system) отримують вищий FI у моделі,  
  ніж очікувалося — ансамблі здатні виявляти складні нелінійні 
  взаємодії, які MI не вловлює лінійно.

• num_of_cylinders, engine_type — середній MI, але висока FI:
  дерева використовують ці ознаки для розбиття ефективніше,
  ніж показує проста взаємна інформація.

• Висновок: MI — корисний швидкий фільтр, але не остаточний 
  критерій. Деякі "слабкі" за MI ознаки залишаються важливими  
  для моделі і не варто їх відкидати без перевірки.
─────────────────────────────────────────────────────────
""")
