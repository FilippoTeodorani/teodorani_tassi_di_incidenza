from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_WEATHER_PATH = BASE_DIR / "data" / "raw" / "usa" / "USA meteo data"
INTERIM_PATH = BASE_DIR / "data" / "interim"

OUTPUT_FILE = INTERIM_PATH / "usa_weather_weekly.csv"


# ======================================================
# HELP FUNCTION
# ======================================================

def compute_relative_humidity(temperature_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    return 100 * (
        np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c)) /
        np.exp((17.625 * temperature_c) / (243.04 + temperature_c))
    )

# ======================================================
# MAIN
# ======================================================ù

def build_weekly_weather_dataset():

    weather_files = list(RAW_WEATHER_PATH.glob("*meteo.csv"))

    if not weather_files:
        raise FileNotFoundError(f"No weather files found in {RAW_WEATHER_PATH}")

    df_list = []

    for file in weather_files:
        df = pd.read_csv(file, parse_dates=["valid_time"])

        region_name = file.stem.replace(" meteo", "").strip()
        df["region"] = region_name

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df["temperature_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15

    df["relative_humidity"] = compute_relative_humidity(
        df["temperature_c"],
        df["dewpoint_c"]
    )

    iso_calendar = df["valid_time"].dt.isocalendar()
    df["year"] = iso_calendar.year
    df["week_number"] = iso_calendar.week

    weekly = (
        df
        .groupby(["region", "year", "week_number"], as_index=False)
        .agg(
            temperature_mean_c=("temperature_c", "mean"),
            dewpoint_mean_c=("dewpoint_c", "mean"),
            relative_humidity_mean=("relative_humidity", "mean")
        )
    )

    weekly["temperature_mean_c"] = weekly["temperature_mean_c"].round(2)
    weekly["dewpoint_mean_c"] = weekly["dewpoint_mean_c"].round(2)
    weekly["relative_humidity_mean"] = weekly["relative_humidity_mean"].round(1)

    weekly = weekly.sort_values(["region", "year", "week_number"])

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(OUTPUT_FILE, index=False)

    print(f"Weekly weather dataset saved to: {OUTPUT_FILE}")
    print(weekly.head())


if __name__ == "__main__":
    build_weekly_weather_dataset()
