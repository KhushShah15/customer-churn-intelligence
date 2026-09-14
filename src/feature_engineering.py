import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH
)


# ==========================================
# LOAD CLEANED DATA
# ==========================================

def load_data():

    file_path = (
        PROCESSED_DATA_PATH
        / "cleaned_customer_data.csv"
    )

    df = pd.read_csv(file_path)

    print("Cleaned dataset loaded successfully!")

    return df


# ==========================================
# CREATE TENURE GROUP
# ==========================================

def create_tenure_group(df):

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 100],
        labels=[
            "New",
            "Growing",
            "Established",
            "Loyal"
        ]
    )

    return df


# ==========================================
# CREATE AVERAGE MONTHLY SPEND
# ==========================================

def create_average_monthly_spend(df):

    # Avoid division by zero
    df["average_monthly_spend"] = (
        df["TotalCharges"]
        /
        df["tenure"].replace(0, 1)
    )

    return df


# ==========================================
# CREATE HIGH MONTHLY CHARGE FLAG
# ==========================================

def create_high_charge_flag(df):

    average_charge = (
        df["MonthlyCharges"].mean()
    )

    df["high_monthly_charge_flag"] = (
        df["MonthlyCharges"]
        > average_charge
    ).astype(int)

    return df


# ==========================================
# CREATE LONG-TERM CONTRACT FLAG
# ==========================================

def create_contract_flag(df):

    df["long_term_contract_flag"] = (
        df["Contract"]
        .isin([
            "One year",
            "Two year"
        ])
    ).astype(int)

    return df


# ==========================================
# CREATE AUTO PAYMENT FLAG
# ==========================================

def create_auto_payment_flag(df):

    auto_payment_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    df["auto_payment_flag"] = (
        df["PaymentMethod"]
        .isin(auto_payment_methods)
    ).astype(int)

    return df


# ==========================================
# CREATE SUPPORT SERVICE COUNT
# ==========================================

def create_support_service_count(df):

    support_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    df["support_services_count"] = (
        df[support_columns]
        .apply(
            lambda row:
            sum(
                row == "Yes"
            ),
            axis=1
        )
    )

    return df


# ==========================================
# CREATE STREAMING SERVICE COUNT
# ==========================================

def create_streaming_service_count(df):

    streaming_columns = [
        "StreamingTV",
        "StreamingMovies"
    ]

    df["streaming_services_count"] = (
        df[streaming_columns]
        .apply(
            lambda row:
            sum(
                row == "Yes"
            ),
            axis=1
        )
    )

    return df


# ==========================================
# CREATE CUSTOMER VALUE SEGMENT
# ==========================================

def create_customer_value_segment(df):

    # Calculate spending quantiles
    q1 = (
        df["TotalCharges"]
        .quantile(0.33)
    )

    q2 = (
        df["TotalCharges"]
        .quantile(0.66)
    )

    def assign_segment(value):

        if value <= q1:
            return "Low Value"

        elif value <= q2:
            return "Medium Value"

        else:
            return "High Value"

    df["customer_value_segment"] = (
        df["TotalCharges"]
        .apply(assign_segment)
    )

    return df


# ==========================================
# CREATE ALL FEATURES
# ==========================================

def create_features(df):

    print("\nCreating engineered features...")

    df = create_tenure_group(df)

    df = create_average_monthly_spend(df)

    df = create_high_charge_flag(df)

    df = create_contract_flag(df)

    df = create_auto_payment_flag(df)

    df = create_support_service_count(df)

    df = create_streaming_service_count(df)

    df = create_customer_value_segment(df)

    print("Feature engineering completed!")

    return df


# ==========================================
# SAVE FEATURED DATA
# ==========================================

def save_data(df):

    output_path = (
        PROCESSED_DATA_PATH
        / "featured_customer_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nFeatured dataset saved successfully!"
    )

    print(
        f"Location: {output_path}"
    )

    print(
        f"\nFinal Dataset Shape: {df.shape}"
    )

    print(
        "\nNew Features Created:"
    )

    new_features = [
        "tenure_group",
        "average_monthly_spend",
        "high_monthly_charge_flag",
        "long_term_contract_flag",
        "auto_payment_flag",
        "support_services_count",
        "streaming_services_count",
        "customer_value_segment"
    ]

    for feature in new_features:

        print(f"✓ {feature}")


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    df = load_data()

    featured_df = create_features(df)

    save_data(featured_df)