import pandas as pd


# ==============================
# 1. LOAD DATASET
# ==============================

file_path = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\n" + "=" * 60)
print("CUSTOMER CHURN INTELLIGENCE SYSTEM")
print("DATA UNDERSTANDING REPORT")
print("=" * 60)


# ==============================
# 2. DATASET SHAPE
# ==============================

print("\nDATASET SHAPE")
print("-" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ==============================
# 3. DATA TYPES
# ==============================

print("\nDATA TYPES")
print("-" * 60)

print(df.dtypes)


# ==============================
# 4. MISSING VALUES
# ==============================

print("\nMISSING VALUES")
print("-" * 60)

print(df.isnull().sum())


# ==============================
# 5. DUPLICATE RECORDS
# ==============================

print("\nDUPLICATE RECORDS")
print("-" * 60)

print(df.duplicated().sum())


# ==============================
# 6. UNIQUE CUSTOMER IDs
# ==============================

print("\nUNIQUE CUSTOMER IDs")
print("-" * 60)

print(df["customerID"].nunique())


# ==============================
# 7. CHURN DISTRIBUTION
# ==============================

print("\nCHURN DISTRIBUTION")
print("-" * 60)

print(df["Churn"].value_counts())


# ==============================
# 8. CHURN PERCENTAGE
# ==============================

print("\nCHURN PERCENTAGE")
print("-" * 60)

print(
    (
        df["Churn"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)