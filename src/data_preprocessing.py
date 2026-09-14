import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH
)


# ==========================================
# 1. LOAD DATA
# ==========================================

def load_data():

    file_path = (
        RAW_DATA_PATH
        / "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")

    return df


# ==========================================
# 2. CLEAN DATA
# ==========================================

def clean_data(df):

    print("\nStarting data cleaning...")

    # Create a copy to avoid modifying
    # the original dataset
    df = df.copy()

    # --------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicates}")

    df = df.drop_duplicates()

    # --------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------

    df.columns = (
        df.columns
        .str.strip()
    )

    # --------------------------------------
    # FIX TotalCharges
    # --------------------------------------

    # Convert blank spaces into missing values
    df["TotalCharges"] = (
        df["TotalCharges"]
        .replace(" ", pd.NA)
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------

    print("\nMissing values after conversion:")

    print(df.isnull().sum())

    # Fill missing TotalCharges values
    # with 0 because these customers
    # generally have tenure = 0
    df["TotalCharges"] = (
        df["TotalCharges"]
        .fillna(0)
    )

    # --------------------------------------
    # STANDARDIZE TARGET VARIABLE
    # --------------------------------------

    df["Churn"] = (
        df["Churn"]
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0
        })
    )

    # --------------------------------------
    # FINAL VALIDATION
    # --------------------------------------

    print("\nData Cleaning Completed!")

    print(f"Final Dataset Shape: {df.shape}")

    print("\nFinal Data Types:")

    print(df.dtypes)

    return df


# ==========================================
# 3. SAVE CLEANED DATA
# ==========================================

def save_data(df):

    # Create processed folder if needed
    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_PATH
        / "cleaned_customer_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nCleaned dataset saved successfully!")

    print(f"Location: {output_path}")


# ==========================================
# 4. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # Load data
    df = load_data()

    # Clean data
    cleaned_df = clean_data(df)

    # Save cleaned data
    save_data(cleaned_df)