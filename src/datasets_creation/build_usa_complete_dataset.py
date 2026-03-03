from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_PATH = BASE_DIR / "data" / "processed"
INTERIM_PATH = BASE_DIR / "data" / "interim"

HHS_FILE = PROCESSED_PATH / "hhs_dataset.csv"
STATE_FILE = PROCESSED_PATH / "usa_state_dataset.csv"
VAX_FILE = INTERIM_PATH / "vaccination_rsv.csv"

OUTPUT_FILE = PROCESSED_PATH / "usa_complete_dataset.csv"

STATE_TO_HHS = {
    "Connecticut": 1, "Maine": 1, "Massachusetts": 1,
    "New Hampshire": 1, "Rhode Island": 1, "Vermont": 1,
    "New Jersey": 2, "New York": 2, "Puerto Rico": 2,
    "Delaware": 3, "District of Columbia": 3, "Maryland": 3,
    "Pennsylvania": 3, "Virginia": 3, "West Virginia": 3,
    "Alabama": 4, "Florida": 4, "Georgia": 4, "Kentucky": 4,
    "Mississippi": 4, "North Carolina": 4,
    "South Carolina": 4, "Tennessee": 4,
    "Illinois": 5, "Indiana": 5, "Michigan": 5,
    "Minnesota": 5, "Ohio": 5, "Wisconsin": 5,
    "Arkansas": 6, "Louisiana": 6, "New Mexico": 6,
    "Oklahoma": 6, "Texas": 6,
    "Iowa": 7, "Kansas": 7, "Missouri": 7,
    "Nebraska": 7,
    "Colorado": 8, "Montana": 8,
    "North Dakota": 8, "South Dakota": 8,
    "Utah": 8, "Wyoming": 8,
    "Arizona": 9, "California": 9,
    "Hawaii": 9, "Nevada": 9,
    "Alaska": 10, "Idaho": 10,
    "Oregon": 10, "Washington": 10
}


# ======================================================
# HELP FUNCTIONS
# ======================================================

def weighted_mean(x, value_col, weight_col):
    w = x[weight_col]
    v = x[value_col]
    mask = (~v.isna()) & (~w.isna())
    if mask.sum() == 0:
        return np.nan
    return np.average(v[mask], weights=w[mask])


# ======================================================
# MAIN
# ======================================================

def build_usa_complete_dataset():

    hhs_data = pd.read_csv(str(HHS_FILE.resolve()))
    state_data = pd.read_csv(str(STATE_FILE.resolve()))
    vax_data = pd.read_csv(str(VAX_FILE.resolve()))

    for df in [hhs_data, state_data]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)

    if "Vaccination Rate" in hhs_data.columns:
        hhs_data.drop(columns="Vaccination Rate", inplace=True)

    # ----------------------------
    # STATE TO HHS
    # ----------------------------

    state_data["HHS region"] = state_data["region"].map(STATE_TO_HHS)

    state_data = state_data.dropna(subset=["HHS region"]).copy()
    state_data["HHS region"] = state_data["HHS region"].astype(int)

    state65 = state_data[state_data["age_class"] == "65+"].copy()

    # ----------------------------
    # POPULATION
    # ----------------------------

    pop_hhs = (
        state65
        .groupby(["HHS region", "flu_season", "region"])["population"]
        .first()
        .reset_index()
    )

    pop_hhs = (
        pop_hhs
        .groupby(["HHS region", "flu_season"])["population"]
        .sum()
        .reset_index(name="population_65plus")
    )

    area_hhs = (
        state_data
        .groupby(["HHS region", "region"])["area"]
        .first()
        .reset_index()
    )

    area_hhs = (
        area_hhs
        .groupby("HHS region")["area"]
        .sum()
        .reset_index(name="area_hhs")
    )

    state65.rename(columns={'hospitalization_rate_fraction': 'hosp_rate_frac'}, inplace=True)
    state_hhs = (
        state65
        .groupby(["HHS region", "week", "flu_season"])
        .apply(lambda g: pd.Series({

            "temp_mean_C":
                weighted_mean(g, "temp_mean_C", "population"),

            "dewpoint_mean_C":
                weighted_mean(g, "dewpoint_mean_C", "population"),

            "RH_mean":
                weighted_mean(g, "RH_mean", "population"),

            "Hospitalization Rate":
                weighted_mean(g, "Hospitalization Rate", "population"),

            "hosp_rate_frac":
                weighted_mean(g, "hosp_rate_frac", "population"),

        }))
        .reset_index()
    )

    # ----------------------------
    # DATA CLEANING AND NORMALIZATION
    # ----------------------------

    vax_data = vax_data[
        pd.to_numeric(vax_data["HHS region"],
                      errors="coerce").notna()
    ].copy()

    vax_data["HHS region"] = vax_data["HHS region"].astype(int)


    hhs_data["HHS region"] = hhs_data["HHS region"].astype(int)

    for df in [hhs_data, vax_data, state_hhs]:
        df["week"] = df["week"].astype(int)
        df["flu_season"] = df["flu_season"].astype(str)

    # ----------------------------
    # FINAL MERGE
    # ----------------------------

    merged = hhs_data.merge(
        state_hhs,
        on=["HHS region", "week", "flu_season"],
        how="outer",
        suffixes=("_hhs", "_state")
    )

    merged = merged.merge(
        vax_data[
            ["HHS region", "week",
             "flu_season", "Vaccination Rate"]
        ],
        on=["HHS region", "week", "flu_season"],
        how="outer"
    )


    merged = merged.merge(
        pop_hhs,
        on=["HHS region", "flu_season"],
        how="left"
    )

    merged = merged.merge(
        area_hhs,
        on="HHS region",
        how="left"
    )

    merged["density_65plus"] = (
        merged["population_65plus"] /
        merged["area_hhs"]
    )


    for col in ["temp_mean_C", "dewpoint_mean_C", "RH_mean"]:
        merged[col] = (
            merged[f"{col}_hhs"]
            .combine_first(merged[f"{col}_state"])
        )

    merged.drop(columns=[
        "temp_mean_C_hhs", "temp_mean_C_state",
        "dewpoint_mean_C_hhs", "dewpoint_mean_C_state",
        "RH_mean_hhs", "RH_mean_state"
    ], inplace=True)


    def build_year_week(row):

        if pd.isna(row["week"]) or pd.isna(row["flu_season"]):
            return np.nan

        start_year = int(row["flu_season"].split("-")[0])
        end_year = int(row["flu_season"].split("-")[1])

        year = start_year if row["week"] >= 40 else end_year

        return f"{year}-{int(row['week']):02d}"

    merged["year_week"] = merged.apply(
        build_year_week,
        axis=1
    )

    merged = merged.sort_values(
        ["HHS region", "flu_season", "year_week"]
    ).copy()

    print(
        "Duplicate keys:",
        merged.duplicated(
            ["HHS region", "week", "flu_season"]
        ).sum()
    )

    target_seasons = [
        "2023-2024",
        "2024-2025",
        "2025-2026"
    ]

    def fix_series(g):

        if g["flu_season"].iloc[0] not in target_seasons:
            return g

        prev = None
        out = []

        for v in g["Vaccination Rate"]:

            if pd.isna(v):
                v = prev

            if prev is not None and v is not None:
                if not pd.isna(v) and v < prev:
                    v = prev

            out.append(v)
            prev = v

        g["Vaccination Rate"] = out
        return g

    merged = (
        merged.groupby(
            ["HHS region", "flu_season"],
            group_keys=False
        )
        .apply(fix_series)
    )

    merged = merged.sort_index()

    merged.rename(columns={'hospitalization_rate_fraction':'hosp_rate_frac'}, inplace=True)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    merged.to_csv(str(OUTPUT_FILE.resolve()), index=False)

    print("USA complete dataset saved.")
    print(merged.head())


if __name__ == "__main__":
    build_usa_complete_dataset()
