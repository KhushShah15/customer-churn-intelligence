import pandas as pd
import joblib

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH
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
# LOAD CUSTOMER DATA
# ==========================================

def load_customer_data():

    data_path = (
        PROCESSED_DATA_PATH
        / "featured_customer_data.csv"
    )

    df = pd.read_csv(
        data_path
    )

    print("Customer dataset loaded successfully!")

    return df


# ==========================================
# FIND CUSTOMER
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
            f"\nCustomer ID '{customer_id}' "
            f"was not found."
        )

        return None

    return customer


# ==========================================
# PREPARE CUSTOMER FEATURES
# ==========================================

def prepare_customer_features(
    customer
):

    X = customer.drop(
        columns=[
            "customerID",
            "Churn"
        ]
    )

    return X


# ==========================================
# CALCULATE RISK LEVEL
# ==========================================

def get_risk_level(
    probability
):

    if probability >= 0.70:
        return "HIGH RISK"

    elif probability >= 0.40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


# ==========================================
# PREDICT CUSTOMER CHURN
# ==========================================

def predict_churn(
    model,
    customer_features
):

    prediction = model.predict(
        customer_features
    )[0]

    probability = (
        model.predict_proba(
            customer_features
        )[0][1]
    )

    return prediction, probability


# ==========================================
# DISPLAY PREDICTION REPORT
# ==========================================

def display_prediction(
    customer,
    prediction,
    probability
):

    customer_id = (
        customer["customerID"]
        .iloc[0]
    )

    risk_level = (
        get_risk_level(
            probability
        )
    )

    print("\n" + "=" * 60)

    print(
        "CUSTOMER CHURN RISK REPORT"
    )

    print("=" * 60)

    print(
        f"\nCustomer ID: "
        f"{customer_id}"
    )

    print(
        f"Churn Probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk Level: "
        f"{risk_level}"
    )

    print(
        f"\nModel Prediction: "
        f"{'Likely to Churn' if prediction == 1 else 'Likely to Stay'}"
    )

    print("\nCustomer Details:")

    print(
        f"Contract: "
        f"{customer['Contract'].iloc[0]}"
    )

    print(
        f"Tenure: "
        f"{customer['tenure'].iloc[0]} months"
    )

    print(
        f"Monthly Charges: "
        f"${customer['MonthlyCharges'].iloc[0]:.2f}"
    )

    print(
        f"Payment Method: "
        f"{customer['PaymentMethod'].iloc[0]}"
    )

    print("\n" + "=" * 60)


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # Load model
    model = load_model()

    # Load customer data
    df = load_customer_data()

    # Show example customer IDs
    print("\nExample Customer IDs:")

    print(
        df["customerID"]
        .head()
        .tolist()
    )

    # Ask user for customer ID
    customer_id = input(
        "\nEnter Customer ID: "
    ).strip()

    # Find customer
    customer = get_customer_data(
        df,
        customer_id
    )

    if customer is not None:

        # Prepare features
        customer_features = (
            prepare_customer_features(
                customer
            )
        )

        # Predict churn
        prediction, probability = (
            predict_churn(
                model,
                customer_features
            )
        )

        # Display result
        display_prediction(
            customer,
            prediction,
            probability
        )