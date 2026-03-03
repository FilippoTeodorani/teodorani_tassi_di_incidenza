from pathlib import Path
import pandas as pd


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_POPULATION_PATH = BASE_DIR / "data" / "raw" / "usa" / "USA population"
INTERIM_PATH = BASE_DIR / "data" / "interim"

OUTPUT_FILE = INTERIM_PATH / "usa_population.csv"

COLUMNS_TO_DROP = [
    "AGE", "SELECTED AGE CATEGORIES", "5 TO 14 YEARS", "15 TO 17 YEARS",
    "18 TO 24 YEARS", "15 TO 44 YEARS", "16 YEARS AND OVER",
    "18 YEARS AND OVER", "21 YEARS AND OVER", "60 YEARS AND OVER",
    "62 YEARS AND OVER", "65 YEARS AND OVER", "75 YEARS AND OVER",
    "SUMMARY INDICATORS", "MEDIAN AGE (YEARS)",
    "SEX RATIO (MALES PER 100 FEMALES)", "AGE DEPENDENCY RATIO",
    "OLD-AGE DEPENDENCY RATIO", "CHILD DEPENDENCY RATIO",
    "PERCENT ALLOCATED", "SEX", "AGE"
]

AGE_MAPPING = {
    "UNDER 5 YEARS": "0-5",
    "5 TO 9 YEARS": "5-9",
    "10 TO 14 YEARS": "10-14",
    "15 TO 19 YEARS": "15-19",
    "20 TO 24 YEARS": "20-24",
    "25 TO 29 YEARS": "25-29",
    "30 TO 34 YEARS": "30-34",
    "35 TO 39 YEARS": "35-39",
    "40 TO 44 YEARS": "40-44",
    "45 TO 49 YEARS": "45-49",
    "50 TO 54 YEARS": "50-54",
    "55 TO 59 YEARS": "55-59",
    "60 TO 64 YEARS": "60-64",
    "65 TO 69 YEARS": "65-69",
    "70 TO 74 YEARS": "70-74",
    "75 TO 79 YEARS": "75-79",
    "80 TO 84 YEARS": "80-84",
    "85 YEARS AND OVER": "85+",
    "TOTAL POPULATION": "total",
}


# ======================================================
# MAIN
# ======================================================

def build_population_dataset():

    population_files = sorted(RAW_POPULATION_PATH.glob("USA*.csv"))

    if not population_files:
        raise FileNotFoundError(f"No population files found in {RAW_POPULATION_PATH}")

    final_dfs = []

    for file in population_files:

        year = file.stem[-4:]
        flu_season = f"{int(year)-1}-{year}"

        df = pd.read_csv(file)

        df.rename(columns={"Label (Grouping)": "age_class"}, inplace=True)

        df["age_class"] = (
            df["age_class"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df = df[~df["age_class"].isin(COLUMNS_TO_DROP)]

        total_cols = [c for c in df.columns if c.endswith("!!Total!!Estimate")]

        long = df.melt(
            id_vars="age_class",
            value_vars=total_cols,
            var_name="region",
            value_name="population"
        )

        long["region"] = long["region"].str.split("!!").str[0]
        long["age_class"] = long["age_class"].map(AGE_MAPPING)

        long["population_raw"] = long["population"].astype(str)

        is_percent = (
            long["population_raw"].str.contains("%", na=False)
            & (long["age_class"] != "total")
        )

        long.loc[is_percent, "population"] = (
            long.loc[is_percent, "population_raw"]
            .str.replace("%", "", regex=False)
            .astype(float) / 100
        )

        long.loc[~is_percent, "population"] = (
            long.loc[~is_percent, "population_raw"]
            .str.replace(",", "", regex=False)
            .replace({"(X)": None, "*****": None})
            .astype(float)
        )

        if is_percent.any():

            totals = (
                long[long["age_class"] == "total"]
                .set_index("region")["population"]
            )

            long.loc[is_percent, "population"] = long.loc[is_percent].apply(
                lambda row: row["population"] * totals[row["region"]],
                axis=1
            )

        out = (
            long
            .groupby(["region", "age_class"], as_index=False)["population"]
            .sum()
        )

        out["flu_season"] = flu_season
        final_dfs.append(out)

    final_df = pd.concat(final_dfs, ignore_index=True)

    final_df = final_df[["flu_season", "region", "age_class", "population"]]

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Population dataset saved to: {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    build_population_dataset()
