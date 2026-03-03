from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "italy" / "meteo"
INTERIM_PATH = BASE_DIR / "data" / "interim"

OUTPUT_FILE = INTERIM_PATH / "italy_weather_weekly.csv"


# ======================================================
# HELP FUNCTION
# ======================================================

def compute_relative_humidity(T, Td):
    return 100 * (
        np.exp((17.625 * Td) / (243.04 + Td)) /
        np.exp((17.625 * T) / (243.04 + T))
    )


# ======================================================
# MAIN
# ======================================================

def build_italy_weather_dataset():

    weather_files = list(RAW_PATH.glob("meteo_data*.csv"))

    if not weather_files:
        raise FileNotFoundError(
            f"No weather files found in {RAW_PATH}"
        )

    df_list = []

    for file in weather_files:
        df = pd.read_csv(
            str(file.resolve()),
            parse_dates=["valid_time"]
        )

        df["city"] = file.stem.replace("meteo_data", "")

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df["t2m_C"] = df["t2m"] - 273.15
    df["d2m_C"] = df["d2m"] - 273.15


    df["RH"] = compute_relative_humidity(
        df["t2m_C"],
        df["d2m_C"]
    )

    df["year"] = df["valid_time"].dt.year

    df["year_start"] = pd.to_datetime(
        df["year"].astype(str) + "-01-01"
    )

    df["week_number"] = (
        (df["valid_time"] - df["year_start"])
        .dt.days // 7
    ) + 1

    weekly = (
        df
        .groupby(["city", "year", "week_number"], as_index=False)
        .agg(
            temp_mean_C=("t2m_C", "mean"),
            dewpoint_mean_C=("d2m_C", "mean"),
            RH_mean=("RH", "mean")
        )
    )

    weekly["temp_mean_C"] = weekly["temp_mean_C"].round(2)
    weekly["dewpoint_mean_C"] = weekly["dewpoint_mean_C"].round(2)
    weekly["RH_mean"] = weekly["RH_mean"].round(1)


    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(
        str(OUTPUT_FILE.resolve()),
        index=False
    )

    print(f"Italy weather dataset saved to: {OUTPUT_FILE}")
    print(weekly.head())


if __name__ == "__main__":
    build_italy_weather_dataset()
