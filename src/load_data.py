import sqlite3
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_PATH


# ==========================================
# PATHS
# ==========================================

DATABASE_PATH = Path("database/churn_database.db")

DATA_PATH = (
    PROCESSED_DATA_PATH
    / "featured_customer_data.csv"
)


# ==========================================
# LOAD CSV DATA
# ==========================================

def load_csv():

    print("\nLoading featured customer dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Dataset Shape: {df.shape}")

    return df


# ==========================================
# PREPARE DATA FOR SQL
# ==========================================

def prepare_customer_data(df):

    customer_df = df[
        [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "InternetService",
            "Contract",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    ].copy()

    # Rename columns to match SQL table
    customer_df = customer_df.rename(
        columns={
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "PhoneService": "phone_service",
            "InternetService": "internet_service",
            "Contract": "contract",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn"
        }
    )

    print("\nCustomer data prepared for SQL.")

    return customer_df


# ==========================================
# INSERT CUSTOMERS INTO DATABASE
# ==========================================

def insert_customers(customer_df):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    print("\nInserting customers into database...")

    insert_query = """
        INSERT OR REPLACE INTO customers
        (
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            internet_service,
            contract,
            payment_method,
            monthly_charges,
            total_charges,
            churn
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    records = list(
        customer_df.itertuples(
            index=False,
            name=None
        )
    )

    cursor.executemany(
        insert_query,
        records
    )

    connection.commit()

    print(
        f"{len(records)} customer records processed."
    )

    connection.close()


# ==========================================
# VERIFY DATABASE
# ==========================================

def verify_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM customers
        """
    )

    total_customers = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM customers
        WHERE churn = 1
        """
    )

    churned_customers = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM customers
        WHERE churn = 0
        """
    )

    active_customers = cursor.fetchone()[0]

    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION")
    print("=" * 60)

    print(
        f"\nTotal Customers: {total_customers}"
    )

    print(
        f"Churned Customers: {churned_customers}"
    )

    print(
        f"Non-Churned Customers: {active_customers}"
    )

    if total_customers > 0:

        churn_rate = (
            churned_customers
            / total_customers
        ) * 100

        print(
            f"Actual Churn Rate: {churn_rate:.2f}%"
        )

    print("\nSample SQL Records:")

    cursor.execute(
        """
        SELECT
            customer_id,
            contract,
            tenure,
            monthly_charges,
            churn
        FROM customers
        LIMIT 5
        """
    )

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    connection.close()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    df = load_csv()

    customer_df = prepare_customer_data(
        df
    )

    insert_customers(
        customer_df
    )

    verify_database()

    print("\n" + "=" * 60)
    print("CUSTOMER DATA LOADING COMPLETED!")
    print("=" * 60)