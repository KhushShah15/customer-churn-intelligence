import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    REPORTS_PATH
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    model_path = (
        MODELS_PATH
        / "best_churn_model.joblib"
    )

    model = joblib.load(
        model_path
    )

    print("Trained model loaded successfully!")

    return model


# ==========================================
# LOAD FEATURED DATA
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

    return df


# ==========================================
# GET CUSTOMER DATA
# ==========================================

def get_customer_data(
    df,
    customer_id
):

    customer = df[
        df["customerID"] == customer_id
    ]

    if customer.empty:

        print(
            f"\nCustomer ID "
            f"'{customer_id}' was not found."
        )

        return None

    return customer


# ==========================================
# PREPARE FEATURES
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
# CREATE SHAP EXPLANATION
# ==========================================

def explain_customer(
    pipeline,
    X_background,
    customer_features
):

    print("\nCreating SHAP explanation...")

    # Extract preprocessing pipeline
    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    # Extract trained ML model
    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    # Transform background data
    X_background_transformed = (
        preprocessor.transform(
            X_background
        )
    )

    # Transform selected customer
    customer_transformed = (
        preprocessor.transform(
            customer_features
        )
    )

    # Get transformed feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Use a sample as background
    background_size = min(
        500,
        len(X_background_transformed)
    )

    background_data = (
        X_background_transformed[
            :background_size
        ]
    )

    # Create SHAP explainer
    explainer = shap.Explainer(
        model,
        background_data,
        feature_names=feature_names
    )

    # Calculate SHAP values
    shap_values = explainer(
        customer_transformed
    )

    print(
        "SHAP explanation created successfully!"
    )

    return (
        shap_values,
        feature_names
    )


# ==========================================
# CLEAN FEATURE NAMES
# ==========================================

def clean_feature_name(
    feature_name
):

    feature_name = (
        feature_name
        .replace(
            "numerical__",
            ""
        )
        .replace(
            "categorical__",
            ""
        )
    )

    return feature_name


# ==========================================
# GET TOP CHURN FACTORS
# ==========================================

def get_top_factors(
    shap_values,
    feature_names,
    top_n=5
):

    # Get SHAP values for one customer
    values = (
        shap_values.values[0]
    )

    factors = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_Value": values
        }
    )

    # Clean names
    factors["Feature"] = (
        factors["Feature"]
        .apply(
            clean_feature_name
        )
    )

    # Positive values increase churn risk
    risk_increasing = (
        factors[
            factors["SHAP_Value"] > 0
        ]
        .sort_values(
            by="SHAP_Value",
            ascending=False
        )
        .head(top_n)
    )

    # Negative values reduce churn risk
    risk_reducing = (
        factors[
            factors["SHAP_Value"] < 0
        ]
        .sort_values(
            by="SHAP_Value",
            ascending=True
        )
        .head(top_n)
    )

    return (
        risk_increasing,
        risk_reducing
    )


# ==========================================
# DISPLAY EXPLANATION
# ==========================================

def display_explanation(
    customer_id,
    risk_increasing,
    risk_reducing
):

    print("\n" + "=" * 65)

    print(
        "CUSTOMER CHURN EXPLANATION"
    )

    print("=" * 65)

    print(
        f"\nCustomer ID: "
        f"{customer_id}"
    )

    print(
        "\nTOP FACTORS "
        "INCREASING CHURN RISK:"
    )

    if len(risk_increasing) > 0:

        for index, row in (
            risk_increasing
            .iterrows()
        ):

            print(
                f"• {row['Feature']} "
                f"(Impact: "
                f"{row['SHAP_Value']:.4f})"
            )

    else:

        print(
            "No major factors found."
        )

    print(
        "\nTOP FACTORS "
        "REDUCING CHURN RISK:"
    )

    if len(risk_reducing) > 0:

        for index, row in (
            risk_reducing
            .iterrows()
        ):

            print(
                f"• {row['Feature']} "
                f"(Impact: "
                f"{row['SHAP_Value']:.4f})"
            )

    else:

        print(
            "No major factors found."
        )

    print("\n" + "=" * 65)


# ==========================================
# SAVE SHAP CHART
# ==========================================

def save_shap_chart(
    shap_values,
    customer_id
):

    output_folder = (
        REPORTS_PATH
        / "figures"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / f"shap_{customer_id}.png"
    )

    plt.figure()

    shap.plots.bar(
        shap_values[0],
        max_display=10,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSHAP chart saved successfully!"
    )

    print(
        f"Location: "
        f"{output_path}"
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # Load model
    pipeline = load_model()

    # Load dataset
    df = load_data()

    # Prepare all features
    X = prepare_features(df)

    # Show example customer IDs
    print("\nExample Customer IDs:")

    print(
        df["customerID"]
        .head()
        .tolist()
    )

    # Ask for customer ID
    customer_id = input(
        "\nEnter Customer ID: "
    ).strip()

    # Get selected customer
    customer = get_customer_data(
        df,
        customer_id
    )

    if customer is not None:

        # Prepare selected customer features
        customer_features = (
            prepare_features(
                customer
            )
        )

        # Create SHAP explanation
        (
            shap_values,
            feature_names
        ) = explain_customer(
            pipeline,
            X,
            customer_features
        )

        # Get important factors
        (
            risk_increasing,
            risk_reducing
        ) = get_top_factors(
            shap_values,
            feature_names
        )

        # Display explanation
        display_explanation(
            customer_id,
            risk_increasing,
            risk_reducing
        )

        # Save SHAP chart
        save_shap_chart(
            shap_values,
            customer_id
        )