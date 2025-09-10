# build_model.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# -------------------------
# Config
# -------------------------
DATA_PATH = r"C:\Users\sidhartha-BD\Desktop\Acclerattors\dead_stock\ms_retail_deadstock_dataset.csv"
MODEL_OUT = "model_pipeline.joblib"
META_OUT = "model_metadata.json"
FORBIDDEN = {"snapshot_date", "sku_id", "store_id"}  # never pick these
TOP_K = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# Helpers
# -------------------------
def safe_top_features(X, y, feature_names, k=10, forbid=None):
    forbid = forbid or set()
    base_names = [fn.split("__")[0].split("_")[0] for fn in feature_names]
    allowed_bases = [b for b in base_names if b not in forbid]

    allowed_cols = [c for c in X.columns if c in allowed_bases]
    if not allowed_cols:
        raise ValueError("No allowed columns left after excluding forbidden features.")

    X_allowed = X[allowed_cols]

    sel = SelectKBest(mutual_info_classif, k=min(k, X_allowed.shape[1]))
    sel.fit(X_allowed.fillna(-999), y)
    chosen = [allowed_cols[i] for i, keep in enumerate(sel.get_support()) if keep]

    while len(chosen) < k and len(chosen) < len(allowed_cols):
        for c in allowed_cols:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= k:
                break
    return chosen

# -------------------------
# Load data
# -------------------------
print("Loading data from", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# -------------------------
# Fix target column
# -------------------------
target_col = "label_deadstock_90d"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

print("Using target column:", target_col)

df = df[~df[target_col].isna()].copy()
if df[target_col].dtype == object:
    df[target_col] = df[target_col].astype(str)

all_columns = list(df.columns)
feature_cols = [c for c in all_columns if c != target_col and c not in FORBIDDEN]
print(f"Candidate features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[target_col].copy()

if y.dtype == object or y.dtype.name == "category":
    y = pd.factorize(y)[0]

is_classification = True
if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 20:
    is_classification = False
if not is_classification:
    raise RuntimeError("This demo is optimized for classification targets (<=20 unique).")

# -------------------------
# Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------------
# Build preprocessing
# -------------------------
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for c in numeric_cols[:]:
    if X_train[c].nunique() <= 10:
        numeric_cols.remove(c)
        cat_cols.append(c)

print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", cat_pipe, cat_cols)
])

preprocessor.fit(X_train)

feature_names = []
if numeric_cols:
    feature_names.extend(numeric_cols)
if cat_cols:
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

# -------------------------
# Select top features
# -------------------------
X_selector = X_train.copy()
for col in cat_cols:
    X_selector[col] = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit_transform(X_selector[[col]])

selected_base_features = safe_top_features(
    X_selector, y_train, feature_names, k=TOP_K, forbid=FORBIDDEN
)
print("Top selected features:", selected_base_features)

final_feature_cols = [c for c in feature_cols if c in selected_base_features]
while len(final_feature_cols) < TOP_K:
    for c in feature_cols:
        if c not in final_feature_cols and c not in FORBIDDEN:
            final_feature_cols.append(c)
        if len(final_feature_cols) >= TOP_K:
            break

final_numeric = [c for c in final_feature_cols if c in numeric_cols]
final_cat = [c for c in final_feature_cols if c in cat_cols]

final_preprocessor = ColumnTransformer([
    ("num", numeric_pipe, final_numeric),
    ("cat", cat_pipe, final_cat)
])

# -------------------------
# Train model
# -------------------------
clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")

pipe = Pipeline([
    ("pre", final_preprocessor),
    ("clf", clf)
])

param_grid = {
    "clf__n_estimators": [200],
    "clf__max_depth": [6, 8],
    "clf__min_samples_leaf": [2, 4]
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

print("Training model...")
grid.fit(X_train[final_feature_cols], y_train)

best = grid.best_estimator_
print("Best params:", grid.best_params_)

# -------------------------
# Evaluate
# -------------------------
y_pred = best.predict(X_test[final_feature_cols])
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# -------------------------
# Save
# -------------------------
joblib.dump(best, MODEL_OUT)

meta = {
    "target_col": target_col,
    "final_feature_cols": final_feature_cols,
    "top_features_asked": final_feature_cols[:TOP_K],
    "test_accuracy": float(acc),
    "forbidden_columns": list(FORBIDDEN),
    "categorical_features": final_cat,
    "numeric_features": final_numeric,
    "categories_map": {col: df[col].dropna().unique().tolist()[:20] for col in final_cat}  # sample categories
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

print("Model + metadata saved.")
