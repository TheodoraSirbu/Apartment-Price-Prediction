
from pathlib import Path
import pandas as pd

# 1. Stabilim caile fisierelor
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "apartments.csv"
OUTPUT_FILE = BASE_DIR / "data" / "apartments_clean.csv"


def preprocess_data():

    # 2. Citim datasetul original
    df = pd.read_csv(INPUT_FILE)

    print("Dataset incarcat cu succes!")
    print("Dimensiune initiala:", df.shape)

    # 3. Afisam coloanele
    print("\nColoanele datasetului:")
    print(df.columns.tolist())

    # 4. Eliminam spatiile din numele coloanelor
    df.columns = df.columns.str.strip()

    # 5. Verificam valorile lipsa
    print("\nValori lipsa:")
    print(df.isnull().sum())

    # 6. Verificam tipurile de date
    print("\nTipurile coloanelor:")
    print(df.dtypes)

    # 7. Afisam statistici descriptive
    print("\nStatistici descriptive:")
    print(df.describe(include="all").to_string())

    # 8. Verificam valorile din coloana Sector
    print("\nValorile din Sector:")
    print(df["Sector"].value_counts(dropna=False).sort_index())

    # 9. Eliminam randurile complet goale
    df = df.dropna(how="all")

    # 10. Eliminam duplicatele exacte
    df = df.drop_duplicates()

    # 11. Salvam datasetul curatat
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocesare finalizata!")
    print("Dimensiune finala:", df.shape)
    print("Fisier generat:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_data()
