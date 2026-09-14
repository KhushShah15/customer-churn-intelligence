import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH
)


# ==========================================
# PATHS
# ==========================================

DATABASE_PATH = Path(
    "database/churn_database.db"
)

MODEL_PATH = (
    MODELS_PATH
    / "best_churn_model.joblib"
)

DATA_PATH = (
    PROCESSED_DATA_PATH
    / "featured_customer_data.csv"
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    print("\nLoading trained churn model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded successfully!")

    return model


# ==========================================
# LOAD CUSTOMER DATA
# ==========================================

def load_data():

    print("\nLoading featured customer data...")

    df = pd.read_csv(
        DATA_PATH
    )

    print("Customer data loaded successfully!")

    print(
        f"Total Customers: {len(df)}"
    )

    return df


# ==========================================
# PREPARE MODEL FEATURES
# ==========================================

def prepare_features(df):

    X = df.drop(
        columns=[
            "customerID",
            "Churn"
        ]
    )

    return X


# ==========================================
# ASSIGN RISK LEVEL
# ==========================================

def assign_risk_level(
    probability
):

    if probability >= 0.70:

        return "HIGH RISK"

    elif probability >= 0.40:

        return "MEDIUM RISK"

    else:

        return "LOW RISK"


# ==========================================
# GENERATE BATCH PREDICTIONS
# ==========================================

def generate_predictions(
    model,
    df
):

    print(
        "\nGenerating churn predictions..."
    )

    X = prepare_features(
        df
    )

    # Predicted class
    predictions = model.predict(
        X
    )

    # Probability customer churns
    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    results = pd.DataFrame(
        {
            "customer_id":
                df["customerID"],

            "churn_probability":
                probabilities,

            "prediction":
                predictions
        }
    )

    results["risk_level"] = (
        results[
            "churn_probability"
        ]
        .apply(
            assign_risk_level
        )
    )

    results["prediction_date"] = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        "Predictions generated successfully!"
    )

    return results


# ==========================================
# SAVE PREDICTIONS TO SQL
# ==========================================

def save_predictions(
    results
):

    print(
        "\nSaving predictions to database..."
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # Remove old predictions
    #
    # This keeps one current prediction
    # per customer for this portfolio version.
    cursor.execute(
        """
        DELETE FROM customer_predictions
        """
    )

    insert_query = """
        INSERT INTO customer_predictions
        (
            customer_id,
            churn_probability,
            prediction,
            risk_level,
            prediction_date
        )
        VALUES (?, ?, ?, ?, ?)
    """

    records = list(
        results.itertuples(
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
        f"{len(records)} predictions saved!"
    )

    connection.close()


# ==========================================
# VERIFY PREDICTIONS
# ==========================================

def verify_predictions():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    print("\n" + "=" * 65)

    print(
        "BATCH PREDICTION VERIFICATION"
    )

    print("=" * 65)

    # Total predictions

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM customer_predictions
        """
    )

    total = cursor.fetchone()[0]

    print(
        f"\nTotal Predictions: {total}"
    )

    # Risk distribution

    cursor.execute(
        """
        SELECT
            risk_level,
            COUNT(*) AS customers
        FROM customer_predictions
        GROUP BY risk_level
        ORDER BY customers DESC
        """
    )

    risk_results = (
        cursor.fetchall()
    )

    print(
        "\nCUSTOMER RISK DISTRIBUTION:"
    )

    for risk, count in risk_results:

        percentage = (
            count / total * 100
        )

        print(
            f"{risk}: "
            f"{count} customers "
            f"({percentage:.2f}%)"
        )

    # Average predicted risk

    cursor.execute(
        """
        SELECT
            AVG(churn_probability)
        FROM customer_predictions
        """
    )

    avg_probability = (
        cursor.fetchone()[0]
    )

    print(
        "\nAverage Predicted "
        f"Churn Probability: "
        f"{avg_probability * 100:.2f}%"
    )

    # Highest risk customers

    cursor.execute(
        """
        SELECT
            customer_id,
            churn_probability,
            risk_level
        FROM customer_predictions
        ORDER BY churn_probability DESC
        LIMIT 10
        """
    )

    high_risk = (
        cursor.fetchall()
    )

    print(
        "\nTOP 10 HIGHEST-RISK CUSTOMERS:"
    )

    for (
        customer_id,
        probability,
        risk
    ) in high_risk:

        print(
            f"{customer_id} | "
            f"{probability * 100:.2f}% | "
            f"{risk}"
        )

    connection.close()


# ==========================================
# SAVE CSV FOR POWER BI
# ==========================================

def save_powerbi_dataset():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT

            c.customer_id,

            c.gender,

            c.senior_citizen,

            c.partner,

            c.dependents,

            c.tenure,

            c.phone_service,

            c.internet_service,

            c.contract,

            c.payment_method,

            c.monthly_charges,

            c.total_charges,

            c.churn AS actual_churn,

            p.churn_probability,

            p.prediction AS predicted_churn,

            p.risk_level,

            p.prediction_date

        FROM customers AS c

        INNER JOIN customer_predictions AS p

        ON c.customer_id = p.customer_id
    """

    powerbi_df = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    output_path = (
        Path("reports")
        / "powerbi"
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        output_path
        / "customer_churn_dashboard.csv"
    )

    powerbi_df.to_csv(
        file_path,
        index=False
    )

    print(
        "\nPower BI dataset created!"
    )

    print(
        f"Location: {file_path}"
    )

    print(
        f"Power BI Rows: "
        f"{len(powerbi_df)}"
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    model = load_model()

    df = load_data()

    results = generate_predictions(
        model,
        df
    )

    save_predictions(
        results
    )

    verify_predictions()

    save_powerbi_dataset()

    print("\n" + "=" * 65)

    print(
        "BATCH CHURN PREDICTION COMPLETED!"
    )

    print("=" * 65)