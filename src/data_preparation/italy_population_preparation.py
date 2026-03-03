from pathlib import Path
import pandas as pd
import re


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_POP_PATH = BASE_DIR / "data" / "raw" / "italy" / "Italian Population"
INTERIM_POP_PATH = BASE_DIR / "data" / "interim"

OUTPUT_FILE = INTERIM_POP_PATH / "population_region_age_year_2002_2025.csv"

REGIONS = [
    "Abruzzo", "Basilicata", "Calabria", "Campania",
    "Emilia-Romagna", "Friuli-Venezia Giulia", "Lazio",
    "Liguria", "Lombardia", "Marche", "Molise",
    "Piemonte", "Puglia", "Sardegna", "Sicilia",
    "Toscana", "Trentino-Alto Adige", "Umbria",
    "Valle d'Aosta", "Veneto"
]


# ======================================================
# HELP FUNCTIONS
# ======================================================

def normalize(text):
    return (
        str(text)
        .lower()
        .replace("-", " ")
        .replace("'", "")
        .replace("’", "")
        .strip()
    )


def extract_region(filename):
    fn = normalize(filename)
    for region in REGIONS:
        if re.search(rf"\b{normalize(region)}\b", fn):
            return region
    return None


def find_age_column(columns):
    for c in columns:
        if re.search(r"\bet", c.lower()):
            return c
    return None


def find_exact_total_column(columns):
    """
    Trova SOLO la colonna esattamente 'Totale'
    (non Totale maschi / Totale femmine)
    """
    for c in columns:
        if normalize(c) == "totale":
            return c
    return None


# ======================================================
# MAIN
# ======================================================

def build_italy_population_dataset():

    files = list(RAW_POP_PATH.glob("*.csv"))
    print("Files found:", len(files))

    all_data = []

    for file in files:

        fname = file.stem
        region = extract_region(fname)

        if region is None:
            print("Region not recognized:", file.name)
            continue

        df = pd.read_csv(
            str(file.resolve()),
            encoding="latin1"
        )

        df.columns = df.columns.str.strip()

        age_col = find_age_column(df.columns)

        if age_col is None:
            print("Age column not found:", file.name)
            continue

        year_cols = [
            c for c in df.columns
            if re.fullmatch(r"20\d{2}", str(c))
        ]

        if year_cols:

            sex_cols = [
                c for c in df.columns
                if "sesso" in c.lower()
            ]

            if sex_cols:
                sex_col = sex_cols[0]
                df[sex_col] = df[sex_col].astype(str).str.lower()
                df = df[df[sex_col].str.contains("tot", na=False)]

            df_long = df.melt(
                id_vars=[age_col],
                value_vars=year_cols,
                var_name="year",
                value_name="population"
            )

            df_long = df_long.rename(columns={age_col: "age"})
            df_long["year"] = df_long["year"].astype(int)
            df_long["region"] = region

            all_data.append(
                df_long[["region", "year", "age", "population"]]
            )

            continue

        year_match = re.search(r"(20\d{2})", fname)
        year = int(year_match.group(1)) if year_match else None

        if year is None:
            print("Year not found in filename:", file.name)
            continue

        total_col = find_exact_total_column(df.columns)

        if total_col is None:
            print("Exact 'Totale' column not found in:", file.name)
            print("Available columns:", df.columns.tolist())
            continue

        df_out = df[[age_col, total_col]].copy()
        df_out.columns = ["age", "population"]

        df_out["year"] = year
        df_out["region"] = region

        all_data.append(
            df_out[["region", "year", "age", "population"]]
        )

    # ==================================================
    # FINAL MERGE
    # ==================================================

    pop = pd.concat(all_data, ignore_index=True)

    pop["age"] = pop["age"].astype(str).str.strip()
    pop = pop[pop["age"].str.fullmatch(r"\d+")]

    pop["age"] = pop["age"].astype(int)

    pop["population"] = pd.to_numeric(
        pop["population"],
        errors="coerce"
    )

    pop = pop.dropna(subset=["population"])

    print("\nRegions found:", pop["region"].nunique())
    print(sorted(pop["region"].unique()))

    pop.to_csv(
        str(OUTPUT_FILE.resolve()),
        index=False
    )

    print(f"Population dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_italy_population_dataset()
