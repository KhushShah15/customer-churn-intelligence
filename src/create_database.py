import sqlite3
from pathlib import Path


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DATABASE_PATH = Path(
    "database/churn_database.db"
)


# ==========================================
# CREATE DATABASE
# ==========================================

def create_database():

    # Create database folder if needed
    DATABASE_PATH.parent.mkdir(
        exist_ok=True
    )

    # Connect to SQLite database
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    print(
        "\nCreating Customer Churn Database..."
    )

    # ======================================
    # TABLE 1: CUSTOMERS
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (

            customer_id TEXT PRIMARY KEY,

            gender TEXT,

            senior_citizen INTEGER,

            partner TEXT,

            dependents TEXT,

            tenure INTEGER,

            phone_service TEXT,

            internet_service TEXT,

            contract TEXT,

            payment_method TEXT,

            monthly_charges REAL,

            total_charges REAL,

            churn INTEGER

        )
        """
    )

    # ======================================
    # TABLE 2: CUSTOMER PREDICTIONS
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_predictions (

            prediction_id INTEGER
            PRIMARY KEY AUTOINCREMENT,

            customer_id TEXT,

            churn_probability REAL,

            prediction INTEGER,

            risk_level TEXT,

            prediction_date TEXT,

            FOREIGN KEY(customer_id)
            REFERENCES customers(customer_id)

        )
        """
    )

    # ======================================
    # TABLE 3: SHAP EXPLANATIONS
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_explanations (

            explanation_id INTEGER
            PRIMARY KEY AUTOINCREMENT,

            customer_id TEXT,

            feature_name TEXT,

            shap_value REAL,

            impact_type TEXT,

            FOREIGN KEY(customer_id)
            REFERENCES customers(customer_id)

        )
        """
    )

    # ======================================
    # TABLE 4: RETENTION RECOMMENDATIONS
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        retention_recommendations (

            recommendation_id INTEGER
            PRIMARY KEY AUTOINCREMENT,

            customer_id TEXT,

            recommendation TEXT,

            priority TEXT,

            FOREIGN KEY(customer_id)
            REFERENCES customers(customer_id)

        )
        """
    )

    # ======================================
    # TABLE 5: MODEL PERFORMANCE
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        model_performance (

            model_id INTEGER
            PRIMARY KEY AUTOINCREMENT,

            model_name TEXT,

            accuracy REAL,

            precision_score REAL,

            recall_score REAL,

            f1_score REAL,

            roc_auc REAL,

            training_date TEXT

        )
        """
    )

    # Save changes
    connection.commit()

    print(
        "\nDatabase created successfully!"
    )

    print(
        f"Location: {DATABASE_PATH}"
    )

    # Show tables
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        """
    )

    tables = cursor.fetchall()

    print(
        "\nDATABASE TABLES:"
    )

    for table in tables:

        print(
            f"✓ {table[0]}"
        )

    # Close connection
    connection.close()

    print(
        "\nDatabase setup completed successfully!"
    )


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    create_database()