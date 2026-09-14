import pandas as pd
import joblib
import shap

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    REPORTS_PATH
)

from src.shap_explainer import (
    prepare_features,
    clean_feature_name
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    model_path = (
        MODELS_PATH
        / "best_churn_model.joblib"
    )

    pipeline = joblib.load(
        model_path
    )

    print("Trained model loaded successfully!")

    return pipeline


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    data_path = (
        PROCESSED_DATA_PATH
        / "featured_customer_data.csv"
    )

    df = pd.read_csv(
        data_path
    )

    print("Featured dataset loaded successfully!")

    print(
        f"Total Customers: {len(df)}"
    )

    return df


# ==========================================
# CREATE SHAP VALUES FOR ALL CUSTOMERS
# ==========================================

def calculate_all_shap_values(
    pipeline,
    df
):

    print(
        "\nPreparing SHAP analysis..."
    )

    # --------------------------------------
    # PREPARE MODEL FEATURES
    # --------------------------------------

    X = prepare_features(
        df
    )

    # --------------------------------------
    # GET PREPROCESSOR AND MODEL
    # --------------------------------------

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    # --------------------------------------
    # TRANSFORM ALL DATA
    # --------------------------------------

    print(
        "Transforming customer features..."
    )

    X_transformed = (
        preprocessor.transform(
            X
        )
    )

    # --------------------------------------
    # FEATURE NAMES
    # --------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    clean_names = [
        clean_feature_name(name)
        for name in feature_names
    ]

    # --------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------

    print(
        "Creating SHAP explainer..."
    )

    background_size = min(
        100,
        len(X_transformed)
    )

    background_data = (
        X_transformed[
            :background_size
        ]
    )

    explainer = shap.Explainer(
        model,
        background_data,
        feature_names=feature_names
    )

    # --------------------------------------
    # CALCULATE SHAP VALUES
    # --------------------------------------

    print(
        "Calculating SHAP values "
        "for all customers..."
    )

    shap_values = explainer(
        X_transformed
    )

    print(
        "SHAP values calculated successfully!"
    )

    return (
        shap_values,
        clean_names
    )


# ==========================================
# CREATE POWER BI SHAP DATASET
# ==========================================

def create_powerbi_shap_dataset(
    df,
    shap_values,
    feature_names,
    top_n=5
):

    print(
        "\nCreating Power BI SHAP dataset..."
    )

    records = []

    customer_ids = (
        df["customerID"]
        .tolist()
    )

    for customer_index, customer_id in enumerate(
        customer_ids
    ):

        values = (
            shap_values.values[
                customer_index
            ]
        )

        customer_factors = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "shap_value":
                    values
            }
        )

        # ----------------------------------
        # RISK-INCREASING FACTORS
        # ----------------------------------

        increasing = (
            customer_factors[
                customer_factors[
                    "shap_value"
                ] > 0
            ]
            .sort_values(
                by="shap_value",
                ascending=False
            )
            .head(top_n)
            .reset_index(
                drop=True
            )
        )

        for rank, row in increasing.iterrows():

            records.append(
                {
                    "customer_id":
                        customer_id,

                    "feature":
                        row["feature"],

                    "shap_value":
                        row["shap_value"],

                    "absolute_shap_value":
                        abs(
                            row["shap_value"]
                        ),

                    "impact_direction":
                        "INCREASES RISK",

                    "rank":
                        rank + 1
                }
            )

        # ----------------------------------
        # RISK-REDUCING FACTORS
        # ----------------------------------

        reducing = (
            customer_factors[
                customer_factors[
                    "shap_value"
                ] < 0
            ]
            .sort_values(
                by="shap_value",
                ascending=True
            )
            .head(top_n)
            .reset_index(
                drop=True
            )
        )

        for rank, row in reducing.iterrows():

            records.append(
                {
                    "customer_id":
                        customer_id,

                    "feature":
                        row["feature"],

                    "shap_value":
                        row["shap_value"],

                    "absolute_shap_value":
                        abs(
                            row["shap_value"]
                        ),

                    "impact_direction":
                        "REDUCES RISK",

                    "rank":
                        rank + 1
                }
            )

        # Progress update
        if (
            customer_index + 1
        ) % 500 == 0:

            print(
                f"Processed "
                f"{customer_index + 1} "
                f"customers..."
            )

    shap_df = pd.DataFrame(
        records
    )

    print(
        "\nPower BI SHAP dataset created!"
    )

    print(
        f"Total SHAP rows: "
        f"{len(shap_df)}"
    )

    return shap_df


# ==========================================
# SAVE POWER BI SHAP DATA
# ==========================================

def save_shap_dataset(
    shap_df
):

    output_folder = (
        REPORTS_PATH
        / "powerbi"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / "customer_shap_explanations.csv"
    )

    shap_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSHAP Power BI file saved successfully!"
    )

    print(
        f"Location: "
        f"{output_path}"
    )


# ==========================================
# VERIFY DATA
# ==========================================

def verify_shap_data(
    shap_df
):

    print("\n" + "=" * 70)

    print(
        "SHAP DATA VERIFICATION"
    )

    print("=" * 70)

    print(
        f"\nTotal Rows: "
        f"{len(shap_df)}"
    )

    print(
        f"Unique Customers: "
        f"{shap_df['customer_id'].nunique()}"
    )

    print(
        "\nImpact Direction Counts:"
    )

    print(
        shap_df[
            "impact_direction"
        ]
        .value_counts()
    )

    print(
        "\nExample Customer Explanation:"
    )

    example_customer = (
        shap_df[
            "customer_id"
        ]
        .iloc[0]
    )

    example_df = (
        shap_df[
            shap_df[
                "customer_id"
            ] == example_customer
        ]
        .sort_values(
            by=[
                "impact_direction",
                "rank"
            ]
        )
    )

    print(
        example_df.to_string(
            index=False
        )
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # Load model
    pipeline = load_model()

    # Load dataset
    df = load_data()

    # Calculate SHAP
    (
        shap_values,
        feature_names
    ) = calculate_all_shap_values(
        pipeline,
        df
    )

    # Build Power BI dataset
    shap_df = (
        create_powerbi_shap_dataset(
            df,
            shap_values,
            feature_names,
            top_n=5
        )
    )

    # Save CSV
    save_shap_dataset(
        shap_df
    )

    # Verify output
    verify_shap_data(
        shap_df
    )

    print("\n" + "=" * 70)

    print(
        "POWER BI SHAP EXPORT COMPLETED!"
    )

    print("=" * 70)