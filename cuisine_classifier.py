
"""
cuisine_classifier.py

Task: Cuisine Classification (Option B - choose the most frequent cuisine among those listed for each restaurant,
where "most frequent" is determined by global frequency of cuisine types in the dataset).

Outputs saved to /mnt/data/:
 - cuisine_model_rf.joblib         : RandomForestClassifier pipeline
 - cuisine_model_lr.joblib         : LogisticRegression pipeline
 - classification_report.csv       : per-model classification metrics summary
 - confusion_matrix_rf.csv         : RandomForest confusion matrix (as CSV)
 - confusion_matrix_lr.csv         : LogisticRegression confusion matrix (as CSV)
 - label_distribution.csv          : distribution of target labels
 - sample_predictions.csv          : sample test set predictions with true/predicted labels
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

INPUT_CSV = "/mnt/data/Dataset .csv"
OUT_DIR = "/mnt/data"

def choose_most_frequent_cuisine(df):
    # Expand cuisine lists and compute global frequency
    cuisines_series = df["Cuisines"].fillna("Unknown").astype(str)
    # Split by comma and strip
    cuisine_lists = cuisines_series.apply(lambda x: [c.strip() for c in x.split(",") if c.strip() != ""])
    all_cuisines = [c for sub in cuisine_lists for c in sub]
    freq = pd.Series(all_cuisines).value_counts()
    # For each row, pick the cuisine among its list with highest global frequency
    def pick_most_freq(lst):
        if not lst:
            return "Unknown"
        # choose the cuisine with maximum freq (if tie, first occurrence)
        return max(lst, key=lambda c: (freq.get(c,0), -lst.index(c)))
    chosen = cuisine_lists.apply(pick_most_freq)
    return chosen, freq

def preprocess_features(df):
    # Simplify and create features
    df["Cuisines_raw"] = df["Cuisines"].fillna("Unknown").astype(str)
    df["Average Cost for two"] = pd.to_numeric(df["Average Cost for two"], errors="coerce").fillna(df["Average Cost for two"].median())
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)
    # Price range numeric
    if "Price range" in df.columns:
        df["Price_range"] = pd.to_numeric(df["Price range"], errors="coerce").fillna(0).astype(int)
    else:
        df["Price_range"] = 0
    # City categorical cleaned
    if "City" in df.columns:
        df["City_clean"] = df["City"].fillna("Unknown").astype(str)
    else:
        df["City_clean"] = "Unknown"
    return df

def build_pipelines():
    # Column selectors
    numeric_cols = ["Average Cost for two", "Votes", "Price_range"]
    cat_cols = ["City_clean"]
    # TF-IDF for full cuisines text
    tfidf = ("cuisines_tfidf", TfidfVectorizer(max_features=500), "Cuisines_raw")
    # Preprocessor
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=0.01))])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols),
        tfidf
    ], remainder="drop")
    # Classifiers
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="saga")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe_lr = Pipeline(steps=[("preprocessor", preprocessor), ("clf", lr)])
    pipe_rf = Pipeline(steps=[("preprocessor", preprocessor), ("clf", rf)])
    return pipe_lr, pipe_rf

def evaluate_and_save(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # Save report and confusion matrix
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUT_DIR, f"classification_report_{name}.csv"))
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    cm_df.to_csv(os.path.join(OUT_DIR, f"confusion_matrix_{name}.csv"))
    # Return summary metrics
    return {"model": name, "accuracy": acc, "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro}

def main():
    df = pd.read_csv(INPUT_CSV)
    df = preprocess_features(df)
    df["target_cuisine"], cuisine_freq = choose_most_frequent_cuisine(df)
    # Save distribution
    cuisine_freq.to_csv(os.path.join(OUT_DIR, "label_distribution.csv"), header=["count"])
    # Filter to top-K cuisines for manageable classification (optional)
    top_k = 12  # limit to top 12 cuisines to avoid severe class imbalance; others labelled 'Other'
    top_cuisines = cuisine_freq.head(top_k).index.tolist()
    df["target_cuisine_reduced"] = df["target_cuisine"].apply(lambda x: x if x in top_cuisines else "Other")
    # Prepare X, y
    X = df[["Cuisines_raw", "Average Cost for two", "Votes", "Price_range", "City_clean"]].copy()
    y = df["target_cuisine_reduced"].copy()
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # Build pipelines
    pipe_lr, pipe_rf = build_pipelines()
    # Fit Logistic Regression
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, os.path.join(OUT_DIR, "cuisine_model_lr.joblib"))
    # Fit Random Forest
    pipe_rf.fit(X_train, y_train)
    joblib.dump(pipe_rf, os.path.join(OUT_DIR, "cuisine_model_rf.joblib"))
    # Evaluate
    metrics_lr = evaluate_and_save(pipe_lr, X_test, y_test, "logistic_regression")
    metrics_rf = evaluate_and_save(pipe_rf, X_test, y_test, "random_forest")
    # Save summary metrics
    summary = pd.DataFrame([metrics_lr, metrics_rf])
    summary.to_csv(os.path.join(OUT_DIR, "classification_summary_metrics.csv"), index=False)
    # Sample predictions save
    sample = X_test.reset_index(drop=True).copy().loc[:49]
    sample["true"] = y_test.reset_index(drop=True).loc[:49].values
    sample["pred_lr"] = pipe_lr.predict(sample)
    sample["pred_rf"] = pipe_rf.predict(sample)
    sample.to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)
    print("Saved outputs to:", OUT_DIR)
    print("Top classes used:", top_cuisines + ["Other"])
    print("Summary metrics:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
