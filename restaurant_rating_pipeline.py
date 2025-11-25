
"""
restaurant_rating_pipeline.py

Standalone script to preprocess data, train regression models, evaluate them,
and save the best RandomForest pipeline + predictions.

Usage:
    python restaurant_rating_pipeline.py

It expects the input CSV at: /mnt/data/Dataset .csv
Outputs saved to /mnt/data/:
    - rf_pipeline.joblib        : trained RandomForest pipeline
    - model_results.csv         : evaluation metrics for all models
    - feature_importances.csv   : RandomForest feature importances
    - sample_predictions.csv    : sample predictions from the test set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

INPUT_PATH = "/mnt/data/Dataset .csv"
OUT_DIR = "/mnt/data"

def build_and_run():
    df = pd.read_csv(INPUT_PATH)
    target_col = "Aggregate rating"

    # Drop identifier/text columns that are not used
    drop_cols = ["Restaurant ID", "Restaurant Name", "Address", "Locality Verbose"]
    # Ensure columns exist before dropping
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Simplify 'Cuisines' by keeping the first cuisine listed
    if "Cuisines" in df.columns:
        df["Cuisines"] = df["Cuisines"].fillna("Unknown").apply(lambda x: str(x).split(",")[0].strip())

    keep_cols = [c for c in df.columns if c not in drop_cols and c != target_col]

    # Select numeric and categorical columns for modeling
    numeric_cols = [c for c in ["Average Cost for two", "Longitude", "Latitude", "Votes"] if c in df.columns]
    cat_cols = [c for c in ["Country Code", "City", "Locality", "Cuisines", 
                            "Has Online delivery", "Is delivering now", "Switch to order menu", 
                            "Price range", "Rating color", "Rating text"] if c in df.columns]

    X = df[keep_cols].copy()
    y = df[target_col].copy()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=0.01))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append({"model": name, "mse": mse, "rmse": rmse, "r2": r2})
        fitted_pipelines[name] = pipe
        print(f"Trained {name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    results_df.to_csv(os.path.join(OUT_DIR, "model_results.csv"), index=False)

    # Save RandomForest pipeline (best non-linear model we used)
    rf_pipe = fitted_pipelines["RandomForest"]
    joblib.dump(rf_pipe, os.path.join(OUT_DIR, "rf_pipeline.joblib"))

    # Extract feature importances
    # Build feature name list (numeric + one-hot categorical)
    num_features = numeric_cols
    ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_features + cat_features

    importances = rf_pipe.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)

    # Save sample predictions (first 10 from test set)
    sample_pred = X_test.copy().reset_index(drop=True).loc[:9]
    sample_pred["true_rating"] = y_test.reset_index(drop=True).loc[:9].values
    sample_pred["predicted_rating"] = rf_pipe.predict(sample_pred[keep_cols])
    sample_pred.to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)

    print("\\nSaved outputs to:", OUT_DIR)
    print(" - rf_pipeline.joblib")
    print(" - model_results.csv")
    print(" - feature_importances.csv")
    print(" - sample_predictions.csv")

if __name__ == "__main__":
    build_and_run()
