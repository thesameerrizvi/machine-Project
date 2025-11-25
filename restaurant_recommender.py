
"""
restaurant_recommender.py (updated)

Content-based restaurant recommender (Option A user-preference format):
    Cuisine (single string), Price Range (int), City (string)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import os

INPUT_CSV = "/mnt/data/Dataset .csv"

def preprocess(df):
    if "Cuisines" in df.columns:
        df["Cuisines_clean"] = df["Cuisines"].fillna("Unknown").apply(lambda x: str(x).split(",")[0].strip())
    else:
        df["Cuisines_clean"] = "Unknown"

    if "Price range" in df.columns:
        df["Price_range_clean"] = pd.to_numeric(df["Price range"], errors="coerce").fillna(0).astype(int)
    else:
        df["Price_range_clean"] = 0

    if "City" in df.columns:
        df["City_clean"] = df["City"].fillna("Unknown").astype(str)
    else:
        df["City_clean"] = "Unknown"

    return df.reset_index(drop=True)

def build_feature_matrix(df):
    features = df[["Cuisines_clean", "City_clean", "Price_range_clean"]].astype(str).copy()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X = encoder.fit_transform(features)
    feature_names = encoder.get_feature_names_out(["Cuisines_clean", "City_clean", "Price_range_clean"])
    return X, encoder, feature_names

def build_user_vector(encoder, cuisine, price_range, city):
    # Build a user dataframe with exact columns that encoder was fitted on
    cols = list(encoder.feature_names_in_)
    # encoder.feature_names_in_ should be ['Cuisines_clean', 'City_clean', 'Price_range_clean']
    user_row = {}
    for c in cols:
        if c == "Cuisines_clean":
            user_row[c] = cuisine if cuisine is not None else "Unknown"
        elif c == "City_clean":
            user_row[c] = city if city is not None else "Unknown"
        elif c == "Price_range_clean":
            user_row[c] = str(price_range) if price_range is not None else "0"
        else:
            user_row[c] = "Unknown"
    user_df = pd.DataFrame([user_row])
    # Ensure column order matches encoder.feature_names_in_
    user_df = user_df[cols].astype(str)
    user_vec = encoder.transform(user_df)
    return user_vec

def recommend(cuisine, price_range, city, top_n=10, save_csv=False):
    df = pd.read_csv(INPUT_CSV)
    df = preprocess(df)
    X, encoder, feature_names = build_feature_matrix(df)

    user_vec = build_user_vector(encoder, cuisine, price_range, city)

    sims = cosine_similarity(X, user_vec).reshape(-1)
    df["similarity"] = sims

    # Filter by city if present
    if city is not None and city in df["City_clean"].values:
        df_filtered = df[df["City_clean"].str.lower() == str(city).lower()].copy()
        if df_filtered.empty:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    sort_cols = ["similarity"]
    if "Aggregate rating" in df_filtered.columns:
        sort_cols.append("Aggregate rating")
    if "Votes" in df_filtered.columns:
        sort_cols.append("Votes")

    recommended = df_filtered.sort_values(sort_cols, ascending=[False, False, False]).head(top_n)

    display_columns = [c for c in ["Restaurant Name", "Cuisines", "City", "Locality", "Price range", "Average Cost for two", "Aggregate rating", "Votes", "similarity"] if c in recommended.columns]
    rec_display = recommended[display_columns].copy()
    rec_display["similarity"] = rec_display["similarity"].round(4)

    if save_csv:
        out_path = "/mnt/data/recommendations_sample.csv"
        rec_display.to_csv(out_path, index=False)
        print(f"Saved recommendations to: {out_path}")

    return rec_display

def _sample_run():
    cuisine = "North Indian"
    price_range = 3
    city = "New Delhi"
    print(f"Running sample recommendation: Cuisine={cuisine}, Price_range={price_range}, City={city}\\n")
    recs = recommend(cuisine, price_range, city, top_n=10, save_csv=True)
    print(recs.to_string(index=False))

if __name__ == "__main__":
    _sample_run()
