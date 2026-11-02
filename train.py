"""Clean data,
 explore it
 train 3 regression model
   compare them
     save the best.


"""
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from config import (DATA_PATH, FEATURE_NAMES, METRICS_PATH, MODEL_PATH,
                    PLOTS_DIR, RANDOM_STATE, TARGET, TEST_SIZE)


# ---------------------------------------------------------------- 1. LOAD
def load_data():
    if not DATA_PATH.exists():
        sys.exit(f"Dataset not found: {DATA_PATH}\n"
                 " "
                 ".")
    # The UCI file uses ';' as separator. sep=None lets pandas detect ',' too.
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.replace('"', "", regex=False)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------- 2. CLEAN
def clean_data(df):
    needed = FEATURE_NAMES + [TARGET]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        sys.exit(f"Dataset is missing required columns: {missing_cols}")

    df = df[needed].copy()
    for col in needed:                       # make sure everything is numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna()                         # drop rows with missing values
    df = df.drop_duplicates()                # drop exact duplicate rows
    df = df[df[TARGET].between(0, 20)]       # valid grade range
    print(f"Cleaning: {before} -> {len(df)} rows")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 3. EDA
def explore(df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Summary statistics ---")
    print(df.describe().round(2).T[["mean", "std", "min", "max"]])

    corr = df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\n--- Correlation of each feature with G3 ---")
    print(corr.round(3))

    # Plot 1: distribution of final grade
    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET], bins=range(0, 22), color="#4f7cac", edgecolor="white")
    plt.title("Distribution of final grade (G3)")
    plt.xlabel("G3"); plt.ylabel("Students")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "g3_distribution.png", dpi=110); plt.close()

    # Plot 2: study time vs G3 (average)
    plt.figure(figsize=(6, 4))
    means = df.groupby("studytime")[TARGET].mean()
    plt.bar(means.index.astype(str), means.values, color="#4f7cac")
    plt.title("Average G3 by study time"); plt.xlabel("studytime (1-4)"); plt.ylabel("Mean G3")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "studytime_vs_g3.png", dpi=110); plt.close()

    # Plot 3: absences vs G3
    plt.figure(figsize=(6, 4))
    plt.scatter(df["absences"], df[TARGET], alpha=0.5, color="#c4553b")
    plt.title("Absences vs G3"); plt.xlabel("absences"); plt.ylabel("G3")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "absences_vs_g3.png", dpi=110); plt.close()

    # Plot 4: previous grade (G2) vs G3
    plt.figure(figsize=(6, 4))
    plt.scatter(df["G2"], df[TARGET], alpha=0.5, color="#3b8f5c")
    plt.title("Previous grade (G2) vs G3"); plt.xlabel("G2"); plt.ylabel("G3")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "g2_vs_g3.png", dpi=110); plt.close()

    return corr


# ---------------------------------------------------------------- 4. TRAIN
def train_and_compare(df):
    X, y = df[FEATURE_NAMES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test).clip(0, 20)
        results[name] = {"MAE": round(float(mean_absolute_error(y_test, pred)), 3),
                         "R2": round(float(r2_score(y_test, pred)), 3)}
        fitted[name] = model

    print("\n--- Model comparison (test set) ---")
    print(pd.DataFrame(results).T)

    best_name = min(results, key=lambda n: results[n]["MAE"])  # lowest error wins
    print(f"\nBest model: {best_name}")

    # Plot 5: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = list(results)
    axes[0].bar(names, [results[n]["MAE"] for n in names], color="#c4553b")
    axes[0].set_title("MAE (lower is better)")
    axes[1].bar(names, [results[n]["R2"] for n in names], color="#3b8f5c")
    axes[1].set_title("R² (higher is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=110); plt.close()

    # Plot 6: actual vs predicted for the best model
    best_pred = fitted[best_name].predict(X_test).clip(0, 20)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, best_pred, alpha=0.6, color="#4f7cac")
    plt.plot([0, 20], [0, 20], "k--", linewidth=1)
    plt.title(f"Actual vs predicted ({best_name})")
    plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=110); plt.close()

    return fitted[best_name], best_name, results, len(X_train), len(X_test)


# ---------------------------------------------------------------- 5. SAVE
def main():
    df = clean_data(load_data())
    corr = explore(df)
    model, best_name, results, n_train, n_test = train_and_compare(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results,
                   "train_rows": n_train, "test_rows": n_test,
                   "correlation_with_G3": {k: round(float(v), 3) for k, v in corr.items()}},
                  f, indent=2)
    print(f"\nSaved model -> {MODEL_PATH}")
    print("Now run:  python app.py")


if __name__ == "__main__":
    main()
