from pathlib import Path
import pandas as pd


# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INTERIM_PATH = BASE_DIR / "data" / "interim"
RAW_PATH = BASE_DIR / "data" / "raw" / "usa"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

RSV_LAB_FILE = INTERIM_PATH / "usa_rsv_lab.csv"
WEATHER_FILE = INTERIM_PATH / "usa_weather_weekly.csv"
POPULATION_FILE = INTERIM_PATH / "usa_population.csv"
USA_AREA_FILE = RAW_PATH / "USA_area.xlsx"

VACC_PATH = RAW_PATH / "USA vaccination"

OUTPUT_FILE = PROCESSED_PATH / "hhs_dataset.csv"

HHS_MAP = {
    "Connecticut": 1, "Maine": 1, "Massachusetts": 1,
    "New Hampshire": 1, "Rhode Island": 1, "Vermont": 1,
    "New Jersey": 2, "New York": 2, "Puerto Rico": 2,
    "Virgin Islands": 2,
    "Delaware": 3, "District of Columbia": 3,
    "Maryland": 3, "Pennsylvania": 3,
    "Virginia": 3, "West Virginia": 3,
    "Alabama": 4, "Florida": 4, "Georgia": 4,
    "Kentucky": 4, "Mississippi": 4,
    "North Carolina": 4, "South Carolina": 4,
    "Tennessee": 4,
    "Illinois": 5, "Indiana": 5, "Michigan": 5,
    "Minnesota": 5, "Ohio": 5, "Wisconsin": 5,
    "Arkansas": 6, "Louisiana": 6,
    "New Mexico": 6, "Oklahoma": 6, "Texas": 6,
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
# MAIN
# ======================================================

def build_hhs_dataset():

    # =========================
    # RSV DATA
    # =========================

    df = pd.read_csv(str(RSV_LAB_FILE.resolve()))
    usa_area = pd.read_excel(str(USA_AREA_FILE.resolve()))

    usa_area["HHS region"] = usa_area["region"].map(HHS_MAP)

    hhs_area = (
        usa_area
        .groupby("HHS region", as_index=False)["area"]
        .sum()
        .rename(columns={"area": "HHS_area"})
    )

    df = df.rename(columns={'HHS region ': 'HHS region'})
    df = (
        df.groupby(
            ["year_week", "HHS region", "week", "flu_season"],
            as_index=False
        )
        .agg({
            "RSV Detections": "sum",
            "RSV Tests": "sum"
        })
    )

    df["RSV positivity rate"] = df["RSV Detections"] / df["RSV Tests"]

    # =========================
    # POPULATION
    # =========================

    population = pd.read_csv(str(POPULATION_FILE.resolve()))

    population["HHS region"] = population["region"].map(HHS_MAP)

    hhs_pop = (
        population[population["age_class"] == "total"]
        .groupby(["flu_season", "HHS region"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "HHS_population"})
    )


    rsv_with_pop = df.merge(
        hhs_pop,
        on=["flu_season", "HHS region"],
        how="left"
    )

    # =========================
    # WEATHER
    # =========================

    meteo_df = pd.read_csv(str(WEATHER_FILE.resolve()))
    meteo_df.rename(columns={"week_number": "week"}, inplace=True)

    meteo_df["HHS region"] = meteo_df["region"].map(HHS_MAP)

    meteo_hhs = (
        meteo_df
        .groupby(["HHS region", "year", "week"], as_index=False)
        .agg({
            "temperature_mean_c": "mean",
            "dewpoint_mean_c": "mean",
            "relative_humidity_mean": "mean"
        })
    )

    meteo_hhs["year_week"] = (
        meteo_hhs["year"].astype(str)
        + "-"
        + meteo_hhs["week"].astype(str).str.zfill(2)
    )

    rsv_final = rsv_with_pop.merge(
        meteo_hhs[
            ["year_week", "HHS region",
             "temperature_mean_c",
             "dewpoint_mean_c",
             "relative_humidity_mean"]
        ],
        on=["year_week", "HHS region"],
        how="left"
    )

    # =========================
    # VACCINATION
    # =========================

    vaccination_2026 = pd.read_csv(
        str((VACC_PATH / "RSV Vaccination USA 2025-26.csv").resolve())
    )

    vaccination_2026_filtered = vaccination_2026[
        (vaccination_2026["Demographic Level"] == "Overall") &
        (vaccination_2026["Indicator_category_label"] == "Vaccinated")
    ].copy()

    vaccination_2026_filtered.drop(columns=['Vaccine', 'Demographic Level', 'Demographic Name', 'Indicator_label', 'Indicator_category_label', 'CI_Half_width_95pct', 'Suppresion_flag', 'Geography_level'], inplace=True )
    vaccination_2026_filtered["Week_ending"] = pd.to_datetime(vaccination_2026_filtered["Week_ending"])
    vaccination_2026_filtered["week"] = vaccination_2026_filtered["Week_ending"].dt.isocalendar().week

    vaccination_2026_filtered["year_week"] = (
        vaccination_2026_filtered["Week_ending"].dt.isocalendar().year.astype(str)
        + "-"
        + vaccination_2026_filtered["week"].astype(str).str.zfill(2)
    )

    def get_flu_season(date):
        year = date.year
        month = date.month

        if month >= 10:
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"

    vaccination_2026_filtered["flu_season"] = vaccination_2026_filtered["Week_ending"].apply(get_flu_season)

    vaccination_2026_filtered["HHS region"] = vaccination_2026_filtered["Geography_name"].apply(
        lambda x: "United States" if x == "National" else x.replace("Region ", "")
    )


    vaccination_2026_filtered["Unweighted Sample Size"] = (
        vaccination_2026_filtered["Unweighted Sample Size"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )


    vaccination_2026_filtered = (
        vaccination_2026_filtered.groupby(["year_week", "HHS region", "week", "flu_season"])
        .apply(lambda g: pd.Series({
            "Estimate": (g["Estimate"] * g["Unweighted Sample Size"]).sum()
                        / g["Unweighted Sample Size"].sum(),
            "Unweighted Sample Size": g["Unweighted Sample Size"].sum()
        }))
        .reset_index()
)

    vaccination_2026 = vaccination_2026_filtered[[
        "year_week",
        "HHS region",
        "week",
        "flu_season",
        "Estimate"
    ]]
    vaccination_2026.rename(columns={'Estimate': 'Vaccination Rate'}, inplace=True)


    vaccination_2025 = pd.read_csv(
        str((VACC_PATH / "RSV Vaccination USA 2024-25.csv").resolve())
    )
    vaccination_2025.drop(columns=['Vaccine','CI_Half_width_90pct', 'Demographic Name', 'Indicator_label', 'CI_Half_width_95pct', 'Suppresion_flag', 'Geography_level'], inplace=True )

    vaccination_2025_filtered = vaccination_2025[
        (vaccination_2025["Demographic Level"] == "Overall") &
        (vaccination_2025["Indicator_category_label"]
         == "Received a vaccination")
    ].copy()

    vaccination_2025_filtered["Week_ending"] = pd.to_datetime(
        vaccination_2025_filtered["Week_ending"]
    )

    vaccination_2025_filtered["week"] = (
        vaccination_2025_filtered["Week_ending"]
        .dt.isocalendar().week.astype(int)
    )

    iso_year = (
        vaccination_2025_filtered["Week_ending"]
        .dt.isocalendar().year.astype(int)
    )

    vaccination_2025_filtered["year_week"] = (
        iso_year.astype(str)
        + "-"
        + vaccination_2025_filtered["week"]
        .astype(str).str.zfill(2)
    )

    vaccination_2025_filtered["flu_season"] = (
        vaccination_2025_filtered["Week_ending"]
        .apply(get_flu_season)
    )

    vaccination_2025_filtered["HHS region"] = (
        vaccination_2025_filtered["Geography_name"]
        .apply(lambda x:
               "United States"
               if x == "National"
               else int(x.replace("Region ", "")))
    )

    vaccination_2025_filtered["Estimate"] = (
        vaccination_2025_filtered["Estimate"] / 100
    )

    vaccination_2025_filtered["weighted_est"] = (
        vaccination_2025_filtered["Estimate"]
        * vaccination_2025_filtered["Unweighted Sample Size"]
    )

    vaccination_2025_grouped = (
        vaccination_2025_filtered
        .groupby(["year_week", "HHS region", "week", "flu_season"],
                 as_index=False)
        .agg(
            weighted_sum=("weighted_est", "sum"),
            sample_sum=("Unweighted Sample Size", "sum")
        )
    )

    vaccination_2025_grouped["Vaccination Rate"] = (
        vaccination_2025_grouped["weighted_sum"]
        / vaccination_2025_grouped["sample_sum"]
    )

    vaccination_2025_grouped = vaccination_2025_grouped[
        ["year_week", "HHS region",
         "week", "flu_season", "Vaccination Rate"]
    ]

    vaccination_2024 = pd.read_csv(
        str((VACC_PATH / "RSV Vaccination USA 2023-24.csv").resolve())
    )

    df_2024 = vaccination_2024.copy()

    # ----------------------------
    # VARIABLE NORMALIZATION
    # ----------------------------

    df_2024["Jurisdiction"] = df_2024["Jurisdiction"].str.strip()

    df_2024["Jurisdiction"] = df_2024["Jurisdiction"].replace({
        "Philadelphia": "Pennsylvania",
        "Pennsylvania (excluding Philadelphia County)": "Pennsylvania",
        "New York City": "New York",
        "New York (excluding New York City)": "New York"
    })


    df_2024["HHS region"] = df_2024["Jurisdiction"].map(HHS_MAP)

    excluded = df_2024[df_2024["HHS region"].isna()]["Jurisdiction"].unique()

    df_2024 = df_2024.dropna(subset=["HHS region"])
    df_2024["HHS region"] = df_2024["HHS region"].astype(int)

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7,
        "AUG": 8, "SEP": 9, "OCT": 10,
        "NOV": 11, "DEC": 12
    }

    df_2024["month_num"] = df_2024["Month"].map(month_map)


    df_2024["flu_season"] = df_2024["Legend"].str.extract(r"(\d{4}-\d{2})")
    df_2024["flu_season"] = df_2024["flu_season"].apply(
        lambda x: x[:5] + "20" + x[-2:]
    )

    df_2024["year"] = df_2024["flu_season"].str[:4].astype(int)

    df_2024["date_ref"] = pd.to_datetime(
        dict(year=df_2024["year"],
             month=df_2024["month_num"],
             day=15)
    )

    df_2024["week"] = df_2024["date_ref"].dt.isocalendar().week.astype(int)

    iso_year = df_2024["date_ref"].dt.isocalendar().year.astype(int)

    df_2024["year_week"] = (
        iso_year.astype(str)
        + "-"
        + df_2024["week"].astype(str).str.zfill(2)
    )


    df_2024["Numerator"] = (
        df_2024["Numerator"]
        .astype(str)
        .str.replace(",", "")
        .replace(
            ["Not Submitted", "Suppressed", "NA", "nan"],
            pd.NA
        )
    )

    df_2024["Numerator"] = pd.to_numeric(
        df_2024["Numerator"],
        errors="coerce"
    )

    df_2024["Population"] = pd.to_numeric(
        df_2024["Population"],
        errors="coerce"
    )

    final_2024 = (
        df_2024.groupby(
            ["year_week", "HHS region", "week", "flu_season"],
            as_index=False
        )
        .agg(
            total_num=("Numerator", "sum"),
            total_pop=("Population", "sum")
        )
    )

    final_2024["Vaccination Rate"] = (
        final_2024["total_num"] / final_2024["total_pop"]
    )

    vaccination_2024_grouped = final_2024[
        ["year_week", "HHS region",
         "week", "flu_season", "Vaccination Rate"]
    ]

    print("\nExcluded jurisdictions (non states):")
    print(list(excluded))

    # =========================
    # CONCATENATION OF ALL VARIABLE
    # =========================

    vaccination_all = pd.concat(
        [
            vaccination_2024_grouped,
            vaccination_2025_grouped,
            vaccination_2026
        ],
        ignore_index=True
    )


    vaccination_all = vaccination_all.sort_values(
        ["flu_season", "HHS region", "week"]
    )
    vaccination_all.to_csv('vaccination_rsv.csv')
    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    VACC_INTERIM_FILE = INTERIM_PATH / "vaccination_rsv.csv"

    vaccination_all.to_csv(
        str(VACC_INTERIM_FILE.resolve()),
        index=False
    )
    # =========================
    # FINAL MERGE
    # =========================

    rsv_final_merged = pd.merge(
        rsv_final,
        vaccination_all,
        on=["year_week", "HHS region", "week", "flu_season"],
        how="left"
    )

    rsv_final_merged = rsv_final_merged.merge(
        hhs_area,
        on="HHS region",
        how="left"
    )

    rsv_final_merged["density"] = (
        rsv_final_merged["HHS_population"] /
        rsv_final_merged["HHS_area"]
    )



    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    rsv_final_merged = rsv_final_merged.rename(columns={
        'temperature_mean_c':"temp_mean_C",
        'dewpoint_mean_c':'dewpoint_mean_C',
        'relative_humidity_mean':'RH_mean',
        'hospitalization_rate':'Hospitalization Rate'
    })
    rsv_final_merged.to_csv(
        str(OUTPUT_FILE.resolve()),
        index=False
    )

    print("HHS dataset saved.")
    print(rsv_final_merged.head())


if __name__ == "__main__":
    build_hhs_dataset()
