import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH,
    REPORTS_PATH
)

from src.ai_recommendations import (
    generate_recommendations,
    get_risk_level
)


# ==========================================
# FILE PATHS
# ==========================================

CUSTOMER_DATA_PATH = (
    PROCESSED_DATA_PATH
    / "featured_customer_data.csv"
)

PREDICTION_DATA_PATH = (
    REPORTS_PATH
    / "powerbi"
    / "customer_churn_dashboard.csv"
)

SHAP_DATA_PATH = (
    REPORTS_PATH
    / "powerbi"
    / "customer_shap_explanations.csv"
)

OUTPUT_PATH = (
    REPORTS_PATH
    / "powerbi"
    / "customer_recommendations.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading customer data...")

    customers = pd.read_csv(
        CUSTOMER_DATA_PATH
    )

    predictions = pd.read_csv(
        PREDICTION_DATA_PATH
    )

    shap_data = pd.read_csv(
        SHAP_DATA_PATH
    )

    print("All datasets loaded successfully!")

    print(
        f"Customers: {len(customers)}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )

    print(
        f"SHAP Rows: {len(shap_data)}"
    )

    return (
        customers,
        predictions,
        shap_data
    )


# ==========================================
# GET CUSTOMER RISK FACTORS
# ==========================================

def get_risk_factors(
    shap_data,
    customer_id
):

    customer_shap = (
        shap_data[
            (
                shap_data["customer_id"]
                == customer_id
            )
            &
            (
                shap_data["impact_direction"]
                == "INCREASES RISK"
            )
        ]
        .sort_values(
            by="rank",
            ascending=True
        )
    )

    risk_factors = (
        customer_shap["feature"]
        .tolist()
    )

    return risk_factors


# ==========================================
# GENERATE ALL RECOMMENDATIONS
# ==========================================

def generate_all_recommendations(
    customers,
    predictions,
    shap_data
):

    print(
        "\nGenerating retention recommendations..."
    )

    records = []

    total_customers = len(customers)

    for index, row in customers.iterrows():

        customer_id = row["customerID"]

        customer = customers[
            customers["customerID"]
            == customer_id
        ]

        prediction_row = (
            predictions[
                predictions["customer_id"]
                == customer_id
            ]
        )

        if prediction_row.empty:
            continue

        churn_probability = (
            prediction_row[
                "churn_probability"
            ]
            .iloc[0]
        )

        risk_level = get_risk_level(
            churn_probability
        )

        risk_factors = (
            get_risk_factors(
                shap_data,
                customer_id
            )
        )

        recommendations = (
            generate_recommendations(
                customer,
                churn_probability,
                risk_factors
            )
        )

        for rec_number, recommendation in enumerate(
            recommendations,
            start=1
        ):

            records.append(
                {
                    "customer_id":
                        customer_id,

                    "churn_probability":
                        churn_probability,

                    "risk_level":
                        risk_level,

                    "recommendation_rank":
                        rec_number,

                    "recommendation":
                        recommendation
                }
            )

        if (
            index + 1
        ) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1} "
                f"of {total_customers} customers..."
            )

    recommendations_df = pd.DataFrame(
        records
    )

    print(
        "\nRecommendations generated successfully!"
    )

    print(
        f"Total Recommendation Rows: "
        f"{len(recommendations_df)}"
    )

    return recommendations_df


# ==========================================
# SAVE FILE
# ==========================================

def save_recommendations(
    recommendations_df
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    recommendations_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nPower BI recommendation file saved!"
    )

    print(
        f"Location: {OUTPUT_PATH}"
    )


# ==========================================
# VERIFY OUTPUT
# ==========================================

def verify_output(
    recommendations_df
):

    print("\n" + "=" * 70)

    print(
        "RETENTION RECOMMENDATION DATA VERIFICATION"
    )

    print("=" * 70)

    print(
        f"\nTotal Rows: "
        f"{len(recommendations_df)}"
    )

    print(
        f"Unique Customers: "
        f"{recommendations_df['customer_id'].nunique()}"
    )

    print(
        "\nRisk Level Distribution:"
    )

    customer_risks = (
        recommendations_df[
            [
                "customer_id",
                "risk_level"
            ]
        ]
        .drop_duplicates()
    )

    print(
        customer_risks[
            "risk_level"
        ]
        .value_counts()
    )

    example_customer = "7590-VHVEG"

    example = (
        recommendations_df[
            recommendations_df[
                "customer_id"
            ] == example_customer
        ]
        .sort_values(
            by="recommendation_rank"
        )
    )

    print(
        f"\nExample Recommendations "
        f"for {example_customer}:"
    )

    if not example.empty:

        for _, row in example.iterrows():

            print(
                f"{row['recommendation_rank']}. "
                f"{row['recommendation']}"
            )

    else:

        print(
            "Example customer not found."
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    (
        customers,
        predictions,
        shap_data
    ) = load_data()

    recommendations_df = (
        generate_all_recommendations(
            customers,
            predictions,
            shap_data
        )
    )

    save_recommendations(
        recommendations_df
    )

    verify_output(
        recommendations_df
    )

    print("\n" + "=" * 70)

    print(
        "POWER BI RECOMMENDATION EXPORT COMPLETED!"
    )

    print("=" * 70)