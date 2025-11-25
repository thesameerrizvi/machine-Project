
"""
location_analysis.py

Task 4: Location‑based Geographical Analysis of Restaurants

Features included:
- Load dataset
- Visualize restaurant distribution by latitude & longitude
- Group by city/locality & compute:
    * Restaurant counts
    * Average ratings
    * Average price range
    * Most common cuisine
- Save all outputs as CSV + plots

Outputs generated in /mnt/data:
- location_scatter_plot.png
- city_summary_stats.csv
- locality_summary_stats.csv
- cuisine_top_by_city.csv
- location_insights.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_CSV = "/mnt/data/Dataset .csv"
OUT_DIR = "/mnt/data"

def load_data():
    df = pd.read_csv(INPUT_CSV)
    return df

def clean_data(df):
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Aggregate rating"] = pd.to_numeric(df["Aggregate rating"], errors="coerce")
    df["Price range"] = pd.to_numeric(df["Price range"], errors="coerce")
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    df["City"] = df["City"].fillna("Unknown")
    df["Locality"] = df["Locality"].fillna("Unknown")
    return df

def make_scatter_plot(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Longitude"], df["Latitude"], s=5)
    plt.title("Restaurant Locations (Longitude vs Latitude)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    out_path = os.path.join(OUT_DIR, "location_scatter_plot.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def summarize_by_city(df):
    summary = df.groupby("City").agg(
        restaurant_count=("Restaurant Name", "count"),
        avg_rating=("Aggregate rating", "mean"),
        avg_price=("Price range", "mean")
    ).reset_index()

    # Most common cuisine per city
    cuisine_list = []
    for city, subset in df.groupby("City"):
        cuisines = subset["Cuisines"].str.split(",").explode().str.strip()
        top = cuisines.value_counts().idxmax() if not cuisines.empty else "Unknown"
        cuisine_list.append([city, top])

    cuisine_df = pd.DataFrame(cuisine_list, columns=["City", "most_common_cuisine"])

    merged = summary.merge(cuisine_df, on="City", how="left")

    merged.to_csv(os.path.join(OUT_DIR, "city_summary_stats.csv"), index=False)
    cuisine_df.to_csv(os.path.join(OUT_DIR, "cuisine_top_by_city.csv"), index=False)

    return merged, cuisine_df

def summarize_by_locality(df):
    summary = df.groupby("Locality").agg(
        restaurant_count=("Restaurant Name", "count"),
        avg_rating=("Aggregate rating", "mean"),
        avg_price=("Price range", "mean")
    ).reset_index()

    summary.to_csv(os.path.join(OUT_DIR, "locality_summary_stats.csv"), index=False)
    return summary

def generate_insights(city_summary, locality_summary):
    insights = []

    # Most restaurant‑dense city
    top_city = city_summary.sort_values("restaurant_count", ascending=False).head(1)
    insights.append(f"City with most restaurants: {top_city['City'].values[0]} ({top_city['restaurant_count'].values[0]})")

    # Highest rated city
    best_rated = city_summary.sort_values("avg_rating", ascending=False).head(1)
    insights.append(f"Highest average rating city: {best_rated['City'].values[0]} ({best_rated['avg_rating'].values[0]:.2f})")

    # Most expensive locality
    expensive_loc = locality_summary.sort_values("avg_price", ascending=False).head(1)
    insights.append(f"Most expensive locality: {expensive_loc['Locality'].values[0]} (Avg Price Range {expensive_loc['avg_price'].values[0]:.2f})")

    # Save insights
    out_path = os.path.join(OUT_DIR, "location_insights.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(insights))

    return out_path

def main():
    df = load_data()
    df = clean_data(df)

    print("Creating scatter plot...")
    scatter_path = make_scatter_plot(df)
    print("Scatter plot saved:", scatter_path)

    print("Summarizing by city...")
    city_summary, cuisine_city = summarize_by_city(df)

    print("Summarizing by locality...")
    locality_summary = summarize_by_locality(df)

    print("Generating insights...")
    insight_path = generate_insights(city_summary, locality_summary)
    print("Insights saved to:", insight_path)

    print("\nTask 4 Complete!")
    print("Files created in /mnt/data:")
    print(" - location_scatter_plot.png")
    print(" - city_summary_stats.csv")
    print(" - locality_summary_stats.csv")
    print(" - cuisine_top_by_city.csv")
    print(" - location_insights.txt")

if __name__ == "__main__":
    main()
