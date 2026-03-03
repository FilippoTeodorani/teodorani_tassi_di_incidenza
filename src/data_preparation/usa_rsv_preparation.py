from pathlib import Path
import pandas as pd


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_RSV_PATH = BASE_DIR / "data" / "raw" / "usa" / "USA RSV"
INTERIM_PATH = BASE_DIR / "data" / "interim"

LAB_OUTPUT = INTERIM_PATH / "usa_rsv_lab.csv"
RATES_OUTPUT = INTERIM_PATH / "usa_rsv_rates.csv"



FLU_CALENDAR = {
    "2011-2012": {"start": 1,  "end": 17, "week_53": False},
    "2012-2013": {"start": 44, "end": 17, "week_53": False},
    "2013-2014": {"start": 43, "end": 17, "week_53": False},
    "2014-2015": {"start": 45, "end": 17, "week_53": False},
    "2015-2016": {"start": 44, "end": 16, "week_53": True},
    "2016-2017": {"start": 44, "end": 17, "week_53": False},
    "2017-2018": {"start": 44, "end": 17, "week_53": False},
    "2018-2019": {"start": 43, "end": 17, "week_53": False},
    "2019-2020": {"start": 44, "end": 17, "week_53": False},
    "2020-2021": {"start": 49, "end": 16, "week_53": True},
    "2021-2022": {"start": 44, "end": 17, "week_53": False},
    "2022-2023": {"start": 43, "end": 17, "week_53": False},
    "2023-2024": {"start": 43, "end": 17, "week_53": False},
    "2024-2025": {"start": 42, "end": 17, "week_53": False},
    "2025-2026": {"start": 42, "end": 52, "week_53": False},
}


def assign_flu_season(year: int, week: int):
    for season, info in FLU_CALENDAR.items():
        start_year, end_year = map(int, season.split("-"))
        start = info["start"]
        end = info["end"]
        allow_53 = info["week_53"]

        if week == 53 and not allow_53:
            continue

        if year == start_year and week >= start:
            return season

        if year == end_year and week <= end:
            return season

    return None


# ======================================================
# LAB DATA PROCESSING
# ======================================================

def process_lab_data():

    lab_file = next(RAW_RSV_PATH.glob("Respiratory_Syncytial_Virus_Laboratory_Data_*.csv"))

    df = pd.read_csv(lab_file)

    df["week_ending_date"] = pd.to_datetime(
        df["Week ending Date"],
        format="%d%b%Y",
        errors="coerce"
    )

    iso = df["week_ending_date"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)

    df["flu_season"] = df.apply(
        lambda r: assign_flu_season(r["year"], r["week"]),
        axis=1
    )

    df = df[df["flu_season"].notna()].reset_index(drop=True)

    df["year_week"] = df.apply(
        lambda r: f"{r['year']}-{r['week']:02d}",
        axis=1
    )

    df = df.rename(columns={"HHS region": "region"})

    df["rsv_positivity_rate"] = df["RSV Detections"] / df["RSV Tests"]

    df = df.drop(columns=[
        "Week ending Date",
        "week_ending_date",
        "year"
    ])

    return df


# ======================================================
# RATES DATA PROCESSING
# ======================================================

def fix_flu_season_format(season: str):
    start, end = season.split("-")
    if len(end) == 2:
        end = "20" + end
    return f"{start}-{end}"


def process_rates_data():

    rates_file = next(RAW_RSV_PATH.glob(
        "Weekly_Rates_of_Laboratory_Confirmed_RSV_Hospitalizations.csv"
    ))

    print(rates_file)

    df = pd.read_csv(rates_file)


    df["week_ending_date"] = pd.to_datetime(
        df["Week ending date"],
        errors="coerce"
    )

    iso = df["week_ending_date"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)

    df["flu_season"] = df["Season"].apply(fix_flu_season_format)

    df["year_week"] = df.apply(
        lambda r: f"{r['year']}-{r['week']:02d}",
        axis=1
    )

    df["age_class"] = df["Age Category"]
    df = df[df["age_class"].notna()]

    df = df.rename(columns={"State": "region"})

    df["valid_season"] = df.apply(
        lambda r: assign_flu_season(r["year"], r["week"]) == r["flu_season"],
        axis=1
    )

    df = df[df["valid_season"]].drop(columns="valid_season")

    df = df.drop(columns=[
        "Season",
        "Week ending date",
        "week_ending_date",
        "Age Category",
        "year"
    ])

    age_classes_keep = [
        "≥18 years (Adults)",
        "0-17 years (Children)",
        "All",
        "18-49 years",
        "50-64 years",
        "≥65 years",
        "0-4 years"
    ]

    age_map = {
        "≥18 years (Adults)": "18+",
        "0-17 years (Children)": "0-17",
        "All": "total",
        "18-49 years": "18-49",
        "50-64 years": "50-64",
        "≥65 years": "65+",
        "0-4 years": "0-4"
    }

    df = df[df["age_class"].isin(age_classes_keep)].copy()
    df["age_class"] = df["age_class"].map(age_map)

    df = df.sort_values(
        ["region", "flu_season", "week", "age_class"]
    ).reset_index(drop=True)

    return df


# ======================================================
# MAIN
# ======================================================

def build_rsv_datasets():

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    lab_df = process_lab_data()
    rates_df = process_rates_data()

    lab_df.to_csv(LAB_OUTPUT, index=False)
    rates_df.to_csv(RATES_OUTPUT, index=False)
    
    print(f"Lab dataset saved to: {LAB_OUTPUT}")
    print(f"Rates dataset saved to: {RATES_OUTPUT}")


if __name__ == "__main__":
    build_rsv_datasets()
