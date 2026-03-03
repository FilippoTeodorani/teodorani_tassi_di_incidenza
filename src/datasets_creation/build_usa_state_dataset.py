from pathlib import Path
import pandas as pd


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INTERIM_PATH = BASE_DIR / "data" / "interim"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

WEATHER_FILE = INTERIM_PATH / "usa_weather_weekly.csv"
POPULATION_FILE = INTERIM_PATH / "usa_population.csv"
RSV_LAB_FILE = INTERIM_PATH / "usa_rsv_lab.csv"
RSV_RATES_FILE = INTERIM_PATH / "usa_rsv_rates.csv"
USA_AREA_FILE = BASE_DIR / "data" / "raw" / "usa" / "USA_area.xlsx"


OUTPUT_FILE = PROCESSED_PATH / "usa_state_dataset.csv"


# ======================================================
# HELP FUNCTIONS
# ======================================================

def compute_flu_season(year, week):
    if week > 40:
        return f"{year}-{year+1}"
    elif week < 20:
        return f"{year-1}-{year}"
    else:
        return None


def map_population_age(age):

    if age == "total":
        return "total"

    if "+" in age:
        return "65+"

    start, end = age.split("-")
    start = int(start)

    if start <= 4:
        return "0-5"
    elif 5 <= start <= 14:
        return "5-14"
    elif 15 <= start <= 64:
        return "15-64"
    else:
        return "65+"


def map_rsv_age(age):

    if age in ["0-4"]:
        return "0-5"
    elif age in ["5-14"]:
        return "5-14"
    elif age in ["18-49", "50-64"]:
        return "15-64"
    elif age == "65+":
        return "65+"
    elif age == "total":
        return "total"
    else:
        return None


# ======================================================
# MAIN
# ======================================================

def build_usa_state_dataset():

    weather = pd.read_csv(str(WEATHER_FILE.resolve()))
    population = pd.read_csv(str(POPULATION_FILE.resolve()))
    rsv_rates = pd.read_csv(str(RSV_RATES_FILE.resolve()))
    usa_area = pd.read_excel(str(USA_AREA_FILE.resolve()))


    # ----------------------------
    # WEATHER
    # ----------------------------

    weather.rename(columns={"week_number": "week"}, inplace=True)

    region_fix = {
        "Massachussets": "Massachusetts",
        "Mississipi": "Mississippi",
        "Okhlaoma": "Oklahoma",
        "Sotuh Dakota": "South Dakota",
        "Winsconsin": "Wisconsin"
    }

    weather["region"] = weather["region"].replace(region_fix)

    us_rows = (
        weather
        .groupby(["year", "week"], as_index=False)
        .agg({
            "temperature_mean_c": "mean",
            "dewpoint_mean_c": "mean",
            "relative_humidity_mean": "mean"
        })
    )

    us_rows["region"] = "United States"
    us_rows = us_rows[weather.columns]

    weather = pd.concat([weather, us_rows], ignore_index=True)

    weather["flu_season"] = weather.apply(
        lambda r: compute_flu_season(r["year"], r["week"]),
        axis=1
    )

    weather = weather.dropna(subset=["flu_season"])
    weather = weather[weather["flu_season"] >= "2017-2018"]

    final_age_classes = ["0-5", "5-14", "15-64", "65+"]

    weather = (
        weather
        .assign(key=1)
        .merge(pd.DataFrame({"age_class": final_age_classes, "key": 1}),
               on="key")
        .drop(columns="key")
    )

    # ----------------------------
    # POPULATION
    # ----------------------------

    population["age_class"] = population["age_class"].apply(map_population_age)

    pop_agg = (
        population
        .groupby(["flu_season", "region", "age_class"], as_index=False)
        .agg({"population": "sum"})
    )

    # ----------------------------
    # RSV DATA
    # ----------------------------

    rsv = rsv_rates.copy()

    rsv = rsv[rsv["Sex"] == "All"].drop(columns=["Sex"])
    rsv = rsv[~rsv["age_class"].isin(["18+"])]

    rsv["region"] = rsv["region"].replace({
        "RSV-NET": "United States"
    })

    rsv["age_class"] = rsv["age_class"].map(map_rsv_age)
    rsv = rsv.dropna(subset=["age_class"])

    # ----------------------------
    # FINAL MERGE
    # ----------------------------

    df = weather.merge(
        rsv,
        on=["region", "week", "age_class", "flu_season"],
        how="left"
    )

    df = df.merge(
        pop_agg,
        on=["region", "age_class", "flu_season"],
        how="left"
    )
    df = df.merge(
        usa_area,
        on="region",
        how="left"
    )

    df["density"] = df["population"] / df["area"]

    df = df.rename(columns={"Rate": "hospitalization_rate"})

    df["hospitalization_rate_fraction"] = df["hospitalization_rate"] / 100_000

    df = df.rename(columns={
        'temperature_mean_c':"temp_mean_C",
        'dewpoint_mean_c':'dewpoint_mean_C',
        'relative_humidity_mean':'RH_mean',
        'hospitalization_rate':'Hospitalization Rate'
    })


    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"USA state dataset saved to: {OUTPUT_FILE}")
    print(list(df.columns))
    print(df.head())


if __name__ == "__main__":
    build_usa_state_dataset()


