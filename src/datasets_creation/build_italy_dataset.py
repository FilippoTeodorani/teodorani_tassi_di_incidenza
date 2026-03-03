from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "italy"
INTERIM_PATH = BASE_DIR / "data" / "interim"
PROCESSED_PATH = BASE_DIR / "data" / "processed"



regional_cases_path = (
    RAW_PATH /
    "InfluNet" /
    "influnet-main" /
    "data-aggregated" /
    "epidemiological_data" /
    "regional_cases.csv"
)

regional_cases = pd.read_csv(str(regional_cases_path.resolve()))

vaccinations_path_regional = (
    RAW_PATH /
    "InfluNet" /
    "influnet-main" /
    "data-aggregated" /
    "vaccinations_data" /
    "timeseries_regional_vaccinations.csv"
)

timeseries_regional_vaccinations = pd.read_csv(
    str(vaccinations_path_regional.resolve())
)



national_cases_path = (
    RAW_PATH /
    "InfluNet" /
    "influnet-main" /
    "data-aggregated" /
    "epidemiological_data" /
    "national_cases.csv"
)

national_cases = pd.read_csv(str(national_cases_path.resolve()))


vaccinations_path = (
    RAW_PATH /
    "InfluNet" /
    "influnet-main" /
    "data-aggregated" /
    "vaccinations_data" /
    "timeseries_national_vaccinations.csv"
)

timeseries_national_vaccinations = pd.read_csv(
    str(vaccinations_path.resolve())
)


meteo_path = INTERIM_PATH / "italy_weather_weekly.csv"

meteo_settimanale = pd.read_csv(
    str(meteo_path.resolve())
)


popolazione_path = INTERIM_PATH / "population_region_age_year_2002_2025.csv"

popolazione_regioni_eta_2002_2025 = pd.read_csv(
    str(popolazione_path.resolve())
)


respi_dir = RAW_PATH / "RespiVirNet"

respi_xlsx = list(respi_dir.glob("*.xlsx"))[0]

weekly_data = pd.read_excel(respi_xlsx, sheet_name="weekly_data")
age_class_1 = pd.read_excel(respi_xlsx, sheet_name="age_class")
age_class_2 = pd.read_excel(respi_xlsx, sheet_name="age_class_2")
age_class_and_region = pd.read_excel(
    respi_xlsx,
    sheet_name="age_class_and_region"
)


superficie_path = (
    RAW_PATH /
    "Italian Population" /
    "Superficie.xlsx"
)

superficie_regioni = pd.read_excel(
    str(superficie_path.resolve())
)

def merge_dataset():
    
    # ======================================================
    # INFLUNET
    # ======================================================

    df = regional_cases.copy()
    df["week"] = df["year_week"].str.split("-").str[1].astype(int)

    region_map = {
        "Piedmont": "Piemonte",
        "Aosta Valley": "Valle d'Aosta",
        "Lombardy": "Lombardia",
        "AP Trento": "Trentino-Alto Adige",
        "AP Bolzano": "Trentino-Alto Adige",
        "Veneto": "Veneto",
        "Friuli Venezia Giulia": "Friuli-Venezia Giulia",
        "Liguria": "Liguria",
        "Emilia-Romagna": "Emilia-Romagna",
        "Tuscany": "Toscana",
        "Umbria": "Umbria",
        "Marche": "Marche",
        "Lazio": "Lazio",
        "Abruzzo": "Abruzzo",
        "Molise": "Molise",
        "Campania": "Campania",
        "Apulia": "Puglia",
        "Basilicata": "Basilicata",
        "Calabria": "Calabria",
        "Sicily": "Sicilia",
        "Sardinia": "Sardegna"
    }

    df["region"] = df["region"].replace(region_map)

    count_cols = [
        "number_healthcare_workers",
        "number_cases",
        "population",
        "cases_0-4", "cases_5-14",
        "cases_15-64", "cases_65+"
    ]

    inc_cols = [
        "incidence",
        "inc_0-4", "inc_5-14",
        "inc_15-64", "inc_65+"
    ]


    def weighted_mean(x, value_col, weight_col):
        return (
            (x[value_col] * x[weight_col]).sum()
            / x[weight_col].sum()
        )


    group_cols = ["flu_season", "year_week", "week", "region"]

    rows = []

    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for col in count_cols:
            row[col] = grp[col].sum()

        row["incidence"] = weighted_mean(
            grp, "incidence", "population"
        )

        for age in ["0-4", "5-14", "15-64", "65+"]:
            row[f"inc_{age}"] = weighted_mean(
                grp, f"inc_{age}", "population"
            )
        rows.append(row)

    df_agg = pd.DataFrame(rows)

    age_map = {
        "total": ("number_cases", "incidence"),
        "0-4": ("cases_0-4", "inc_0-4"),
        "5-14": ("cases_5-14", "inc_5-14"),
        "15-64": ("cases_15-64", "inc_15-64"),
        "65+": ("cases_65+", "inc_65+")
    }

    long_dfs = []

    for age_class, (cases_col, inc_col) in age_map.items():

        tmp = df_agg[
            ["flu_season", "year_week", "week", "region"]
        ].copy()

        tmp["age_class"] = age_class
        tmp["cases"] = df_agg[cases_col]
        tmp["incidence"] = df_agg[inc_col]

        long_dfs.append(tmp)

    regional_cases_final = pd.concat(
        long_dfs,
        ignore_index=True
    )

    regional_cases_final = regional_cases_final.sort_values(
        ["region", "year_week", "age_class"]
    )

    df = national_cases.copy()
    df["region"] = "Italy"

    df["week"] = (
        df["year_week"]
        .astype(str)
        .str.split("-")
        .str[1]
        .astype(int)
    )

    sum_cols = [
        "number_healthcare_workers",
        "number_cases",
        "population",
        "cases_0-4", "cases_5-14",
        "cases_15-64", "cases_65+",
        "incidence",
        "inc_0-4", "inc_5-14",
        "inc_15-64", "inc_65+"
    ]

    group_cols = ["flu_season", "year_week", "week", "region"]

    df_agg = (
        df
        .groupby(group_cols, as_index=False)[sum_cols]
        .sum()
    )

    age_map = {
        "total": ("number_cases", "incidence"),
        "0-4": ("cases_0-4", "inc_0-4"),
        "5-14": ("cases_5-14", "inc_5-14"),
        "15-64": ("cases_15-64", "inc_15-64"),
        "65+": ("cases_65+", "inc_65+")
    }

    long_dfs = []
    base_cols = ["flu_season", "year_week", "week", "region"]

    for age_class, (cases_col, inc_col) in age_map.items():

        tmp = df_agg[base_cols].copy()

        tmp["age_class"] = age_class
        tmp["cases"] = df_agg[cases_col]
        tmp["incidence"] = df_agg[inc_col]

        long_dfs.append(tmp)

    national_cases_final = pd.concat(
        long_dfs,
        ignore_index=True
    )

    national_cases_final = (
        national_cases_final
        .sort_values(["region", "year_week", "age_class"])
        .reset_index(drop=True)
    )


    df = timeseries_regional_vaccinations.copy()

    df = df.rename(columns={
        "Region": "region",
        "flu_season": "flu_season",
        "elderly_population": "elderly_population",
        "total_population": "total_population"
    })

    df["region"] = df["region"].replace(region_map)


    def weighted_or_valid_mean(values, weights):
        """
        Weighted mean ignoring NaN.
        If only one valid value exists, return it.
        """
        mask = (~values.isna()) & (~weights.isna())

        if mask.sum() == 0:
            return np.nan

        if mask.sum() == 1:
            return values[mask].iloc[0]

        return np.average(values[mask], weights=weights[mask])


    rows = []

    for (flu_season, region), grp in df.groupby(["flu_season", "region"]):

        row = {
            "flu_season": flu_season,
            "region": region,
            "elderly_population": weighted_or_valid_mean(
                grp["elderly_population"],
                grp["total_population"]
            ),
            "total_population": weighted_or_valid_mean(
                grp["total_population"],
                grp["total_population"]
            )
        }

        rows.append(row)

    vacc_agg = pd.DataFrame(rows)

    season_weeks = (
        regional_cases_final[["flu_season", "week", "region"]]
        .drop_duplicates()
    )

    vacc_expanded = season_weeks.merge(
        vacc_agg,
        on=["flu_season", "region"],
        how="left"
    )

    age_classes = ["total", "0-4", "5-14", "15-64", "65+"]

    long_dfs = []

    for age in age_classes:

        tmp = vacc_expanded[
            ["flu_season", "week", "region"]
        ].copy()

        tmp["age_class"] = age

        if age == "65+":
            tmp["vaccination_ratio"] = vacc_expanded["elderly_population"]

        elif age == "total":
            tmp["vaccination_ratio"] = vacc_expanded["total_population"]

        else:
            tmp["vaccination_ratio"] = np.nan

        long_dfs.append(tmp)

    vaccinations_final = pd.concat(
        long_dfs,
        ignore_index=True
    )

    vaccinations_final = vaccinations_final.sort_values(
        ["region", "flu_season", "week", "age_class"]
    )

    df = timeseries_national_vaccinations.copy()

    df['region'] = 'Italy'

    season_weeks = (
        national_cases_final[["flu_season", "week", "region"]]
        .drop_duplicates()
    )

    vacc_expanded = season_weeks.merge(
        df,
        on=["flu_season", "region"],
        how="left"
    )

    age_classes = ["total", "0-4", "5-14", "15-64", "65+"]

    long_dfs = []

    for age in age_classes:
        tmp = vacc_expanded[["flu_season", "week", "region"]].copy()
        tmp["age_class"] = age
        
        if age == "65+":
            tmp["vaccination_ratio"] = vacc_expanded["elderly_population"]
        elif age =='total':
            tmp["vaccination_ratio"] = vacc_expanded["total_population"]
        
        long_dfs.append(tmp)

    vaccinations_final = pd.concat(long_dfs, ignore_index=True)
    vaccinations_final = vaccinations_final.sort_values(
        ["region", "flu_season", "week", "age_class"]
    )

    # =========================
    # 5. WEATHER
    # =========================

    meteo = meteo_settimanale.copy()
    meteo = meteo.rename(columns={"city": "region"})

    city_to_region = {
        " Ancona": "Marche",
        " Genova": "Liguria",
        " Aosta": "Valle d'Aosta",
        " Milano": "Lombardia",
        " Trento": "Trentino-Alto Adige",
        " Venezia": "Veneto",
        " Trieste": "Friuli-Venezia Giulia",
        " Bologna": "Emilia-Romagna",
        " Firenze": "Toscana",
        " Perugia": "Umbria",
        " Roma": "Lazio",
        " Napoli": "Campania",
        " L'Aquila": "Abruzzo",
        " Camobasso": "Molise",
        " Potenza": "Basilicata",
        " Cosenza": "Calabria",
        " Bari": "Puglia",
        " Palermo": "Sicilia",
        " Cagliari": "Sardegna",
        " Torino": "Piemonte"
    }

    meteo["region"] = meteo["region"].replace(city_to_region)
    meteo = meteo[~meteo["week_number"].between(18, 41)]

    def get_flu_season(row):
        week = row["week_number"]
        year = row["year"]
        if week >= 42:
            return f"{year}-{year+1}"
        else:  # week 1-17
            return f"{year-1}-{year}"

    meteo["flu_season"] = meteo.apply(get_flu_season, axis=1)

    meteo = meteo.sort_values(["region", "flu_season", "week_number"])

    meteo_italy = (
        meteo
        .groupby(["flu_season", "week_number"], as_index=False)
        .agg({
            "temp_mean_C": "mean",
            "dewpoint_mean_C": "mean",
            "RH_mean": "mean"
        })
    )

    meteo_italy["region"] = "Italy"

    meteo_final = pd.concat(
        [meteo, meteo_italy],
        ignore_index=True
    ).sort_values(["region", "flu_season", "week_number"]).reset_index(drop=True)

    meteo_final.rename(columns={'week_number':'week'}, inplace=True)
    meteo_final.drop(columns='year', inplace=True)
    meteo_final.head(12)

    # =========================
    # POPULATION
    # =========================

    pop = popolazione_regioni_eta_2002_2025.copy()
    pop = pop[pop["age"] != "Totale"]

    pop["age"] = pop["age"].replace({"100 e oltre": "100"})

    pop = pop.rename(columns={
        "Regione": "region",
        "Anno": "year",
        "age": "age",
        "Popolazione": "population"
    })

    pop["age"] = pd.to_numeric(pop["age"], errors="coerce")

    def age_class_mapping(age):
        if pd.isna(age):
            return None  
        elif 0 <= age <= 4:
            return "0-4"
        elif 5 <= age <= 14:
            return "5-14"
        elif 15 <= age <= 64:
            return "15-64"
        else:
            return "65+"

    pop["age_class"] = pop["age"].apply(age_class_mapping)

    pop_class = (
        pop.groupby(["region", "year", "age_class"], as_index=False)
        .agg({"population": "sum"})
    )

    season_weeks = (
        meteo_final[["flu_season", "week", "region"]]
        .drop_duplicates()
    )

    age_classes = ["0-4", "5-14", "15-64", "65+"]

    season_weeks_expanded = season_weeks.assign(key=1).merge(
        pd.DataFrame({"age_class": age_classes, "key": [1]*len(age_classes)}),
        on="key"
    ).drop("key", axis=1)

    def get_year_from_flu(row):
        start, end = map(int, row["flu_season"].split("-"))
        if row["week"] >= 42:
            return start
        else:
            return end

    season_weeks_expanded["year"] = season_weeks_expanded.apply(get_year_from_flu, axis=1)
    pop_expanded =pop_class.merge(
        season_weeks_expanded,
        on=["region", "year", "age_class"],
        how="left"
    )

    pop_expanded = pop_expanded.sort_values(
        ["region", "flu_season", "week", "age_class"]
    ).reset_index(drop=True)

    pop_expanded.head(12)

    pop_italy = (
        pop_expanded
        .groupby(["flu_season", "week", "year", "age_class"], as_index=False)
        .agg({"population": "sum"})
    )

    pop_italy["region"] = "Italy"

    pop_total = (
        pd.concat([pop_expanded, pop_italy], ignore_index=True)
        .groupby(["region", "flu_season", "week", "year"], as_index=False)
        .agg({"population": "sum"})
    )

    pop_total["age_class"] = "total"


    pop_final = (
        pd.concat([pop_expanded, pop_italy, pop_total], ignore_index=True)
        .sort_values(["region", "flu_season", "week", "age_class"])
        .reset_index(drop=True)
    )

    pop_final.head()

    # =========================
    # RESPIVIRNET
    # =========================

    weekly = weekly_data.copy()
    weekly = weekly.rename(columns={
        "Season": "flu_season",
        "Week": "week"
    })

    def fix_flu_season(season):
        start, end = season.split("-")
        if len(end) == 2:
            end = "20" + end
        return f"{start}-{end}"

    weekly["flu_season"] = weekly["flu_season"].apply(fix_flu_season)

    cols_to_suffix = [c for c in weekly.columns if c not in ["flu_season", "week"]]
    weekly = weekly.rename(columns={c: f"{c}_national_general_data" for c in cols_to_suffix})

    weekly_backbone = weekly[["flu_season", "week"]].drop_duplicates().copy()
    weekly_backbone["region"] = "Italy"
    weekly_backbone["age_class"] = "total"

    weekly_final = weekly_backbone.merge(
        weekly,
        on=["flu_season", "week"],
        how="left"
    )

    weekly_final = weekly_final.sort_values(
        ["region", "flu_season", "week", "age_class"]
    ).reset_index(drop=True)

    weekly_final.head(12)

    age_data = age_class_1.copy()
    age_data = age_data.rename(columns={
        "Year": "flu_season",
        "Week": "week",
        "Age Class": "age_class"
    })

    def fix_flu_season(season):
        start, end = season.split("-")
        if len(end) == 2:
            end = "20" + end
        return f"{start}-{end}"

    age_data["flu_season"] = age_data["flu_season"].apply(fix_flu_season)

    cols_to_suffix = [
        c for c in age_data.columns
        if c not in ["flu_season", "week", "age_class"]
    ]
    age_data = age_data.rename(
        columns={c: f"{c}_national_data" for c in cols_to_suffix}
    )

    age_backbone = age_data[["flu_season", "week", "age_class"]].drop_duplicates().copy()
    age_backbone["region"] = "Italy"

    age_final = age_backbone.merge(
        age_data,
        on=["flu_season", "week", "age_class"],
        how="left"
    )

    age_final = age_final.sort_values(
        ["region", "flu_season", "week", "age_class"]
    ).reset_index(drop=True)

    age_final.head(12)

    meteo_expanded = meteo_final.assign(key=1).merge(
        pd.DataFrame({"age_class": age_classes, "key": [1]*len(age_classes)}),
        on="key"
    ).drop("key", axis=1)
    meteo_expanded.head(12)

    # =========================
    # AREA
    # =========================

    superficie_regioni.rename(columns={'Regione': 'region'}, inplace=True)

    italy_row = (
        superficie_regioni
        .drop(columns=["region"])
        .sum(numeric_only=True)
        .to_frame()
        .T
    )

    italy_row["region"] = "Italy"

    superficie_regioni_final = pd.concat(
        [superficie_regioni, italy_row],
        ignore_index=True
    )

    superficie_regioni_final

    age_classes = ['total', "0-4", "5-14", "15-64", "65+"]

    meteo_expanded = meteo_final.assign(key=1).merge(
        pd.DataFrame({"age_class": age_classes, "key": [1]*len(age_classes)}),
        on="key"
    ).drop("key", axis=1)

    # =========================
    # FINAL MERGE
    # =========================

    backbone = pd.concat([
        regional_cases_final[["region", "flu_season", "week", "age_class"]],
        pop_final[["region", "flu_season", "week", "age_class"]],
        meteo_expanded[["region", "flu_season", "week", 'age_class']],
        weekly_final[["region", "flu_season", "week", "age_class"]],
        age_final[["region", "flu_season", "week", "age_class"]],
    ], ignore_index=True).drop_duplicates()
    
    df_final = backbone.copy()

    df_final = df_final.merge(
        regional_cases_final,
        on=["region", "flu_season", "week", "age_class"],
        how="left"
    )

    df_final = df_final.merge(
        vaccinations_final,
        on=["region", "flu_season", "week", "age_class"],
        how="left"
    )

    df_final = df_final.merge(
        meteo_expanded,
        on=["region", "flu_season", "week", 'age_class'],
        how="left"
    )

    df_final = df_final.merge(
        pop_final,
        on=["region", "flu_season", "week", "age_class"],
        how="left"
    )

    df_final = df_final.merge(
        weekly_final,
        on=["region", "flu_season", "week", "age_class"],
        how="left"
    )

    df_final = df_final.merge(
        age_final,
        on=["region", "flu_season", "week", "age_class"],
        how="left"
    )
    df_final = df_final.merge(
        superficie_regioni_final,
        on="region",
        how="left"
    )
    df_final = df_final.sort_values(
        ["region", "flu_season", "week", "age_class"]
    ).reset_index(drop=True)

    df_final["density"] = df_final["population"] / df_final["Superficie"]

    df_final = df_final.rename(columns={
        "cases": "cases_influenza",
        "incidence": "incidence_influenza",
        "vaccination_ratio": "vaccination_ratio_influenza"
    })

    print(df_final.shape)
    df_final.head(12)
    flu_calendar = {    
    "2003-2004": {"start": 42, "end": 17, "week_53": False},
    "2004-2005": {"start": 42, "end": 16, "week_53": True},
    "2005-2006": {"start": 42, "end": 16, "week_53": False},
    "2006-2007": {"start": 42, "end": 17, "week_53": False},
    "2007-2008": {"start": 42, "end": 17, "week_53": False},
    "2008-2009": {"start": 42, "end": 17, "week_53": False},
    "2009-2010": {"start": 42, "end": 15, "week_53": True},
    "2010-2011": {"start": 42, "end": 17, "week_53": False},
    "2011-2012": {"start": 42, "end": 17, "week_53": False},
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
    "2025-2026": {"start": 42, "end": 52, "week_53": False}
    }

    def is_valid_flu_week(row):
        season = row["flu_season"]
        week = row["week"]

        if season not in flu_calendar:
            return False  

        info = flu_calendar[season]
        start = info["start"]
        end = info["end"]
        allow_53 = info["week_53"]

        if week == 53 and not allow_53:
            return False

        return (week >= start) or (week <= end)

    df_final = df_final[df_final.apply(is_valid_flu_week, axis=1)]
    df_final = df_final.reset_index(drop=True)
    def build_year_week(row):
        flu_season = row["flu_season"]
        week = int(row["week"])

        start_year, end_year = map(int, flu_season.split("-"))

        if week < 20:
            year = end_year
        else:
            year = start_year

        return f"{year}-{week:02d}"

    df_final["year_week"] = df_final.apply(build_year_week, axis=1)
    df_final.drop(columns=['year'], inplace=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_PATH / "italy_dataset.csv"

    df_final.to_csv(output_file, index=False)

if __name__ == "__main__":
    merge_dataset()