import sqlite3
import pandas as pd
from pathlib import Path


DATABASE_PATH = Path(
    "database/churn_database.db"
)


# ==========================================
# RUN SQL QUERY
# ==========================================

def run_query(query):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    df = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return df


# ==========================================
# BUSINESS KPI ANALYSIS
# ==========================================

def business_kpis():

    query = """
        SELECT

            COUNT(*) AS total_customers,

            SUM(churn) AS churned_customers,

            ROUND(
                100.0 * SUM(churn) / COUNT(*),
                2
            ) AS churn_rate,

            ROUND(
                SUM(monthly_charges),
                2
            ) AS monthly_revenue

        FROM customers
    """

    return run_query(query)


# ==========================================
# RISK DISTRIBUTION
# ==========================================

def risk_distribution():

    query = """
        SELECT

            risk_level,

            COUNT(*) AS customers,

            ROUND(
                AVG(churn_probability) * 100,
                2
            ) AS avg_churn_probability

        FROM customer_predictions

        GROUP BY risk_level

        ORDER BY customers DESC
    """

    return run_query(query)


# ==========================================
# REVENUE AT RISK
# ==========================================

def revenue_at_risk():

    query = """
        SELECT

            COUNT(*) AS high_risk_customers,

            ROUND(
                SUM(c.monthly_charges),
                2
            ) AS monthly_revenue_at_risk

        FROM customers AS c

        INNER JOIN customer_predictions AS p

            ON c.customer_id =
               p.customer_id

        WHERE p.risk_level =
              'HIGH RISK'
    """

    return run_query(query)


# ==========================================
# TOP HIGH-RISK CUSTOMERS
# ==========================================

def top_risk_customers():

    query = """
        SELECT

            c.customer_id,

            c.contract,

            c.tenure,

            c.monthly_charges,

            ROUND(
                p.churn_probability * 100,
                2
            ) AS churn_probability,

            p.risk_level

        FROM customers AS c

        INNER JOIN customer_predictions AS p

            ON c.customer_id =
               p.customer_id

        ORDER BY
            p.churn_probability DESC

        LIMIT 10
    """

    return run_query(query)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CUSTOMER CHURN SQL BUSINESS ANALYSIS")
    print("=" * 70)

    print("\nBUSINESS KPIs")
    print("-" * 70)

    print(
        business_kpis()
        .to_string(index=False)
    )

    print("\nCUSTOMER RISK DISTRIBUTION")
    print("-" * 70)

    print(
        risk_distribution()
        .to_string(index=False)
    )

    print("\nREVENUE AT RISK")
    print("-" * 70)

    print(
        revenue_at_risk()
        .to_string(index=False)
    )

    print("\nTOP 10 HIGH-RISK CUSTOMERS")
    print("-" * 70)

    print(
        top_risk_customers()
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("SQL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)