import pandas as pd


# ==========================================
# DETERMINE CUSTOMER RISK LEVEL
# ==========================================

def get_risk_level(probability):

    if probability >= 0.70:
        return "HIGH RISK"

    elif probability >= 0.40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


# ==========================================
# GENERATE RETENTION RECOMMENDATIONS
# ==========================================

def generate_recommendations(
    customer,
    churn_probability,
    risk_factors
):

    recommendations = []

    contract = customer["Contract"].iloc[0]

    tenure = customer["tenure"].iloc[0]

    monthly_charges = (
        customer["MonthlyCharges"]
        .iloc[0]
    )

    payment_method = (
        customer["PaymentMethod"]
        .iloc[0]
    )

    # --------------------------------------
    # HIGH CHURN RISK
    # --------------------------------------

    if churn_probability >= 0.70:

        recommendations.append(
            "Prioritize immediate retention outreach "
            "because this customer has a high churn risk."
        )

        recommendations.append(
            "Assign the customer to a proactive "
            "customer success or support representative."
        )

    # --------------------------------------
    # MEDIUM CHURN RISK
    # --------------------------------------

    elif churn_probability >= 0.40:

        recommendations.append(
            "Monitor this customer closely and "
            "send a personalized retention offer."
        )

    # --------------------------------------
    # LOW CHURN RISK
    # --------------------------------------

    else:

        recommendations.append(
            "Maintain regular engagement and "
            "continue providing strong customer support."
        )

    # --------------------------------------
    # CONTRACT STRATEGY
    # --------------------------------------

    if contract == "Month-to-month":

        recommendations.append(
            "Offer a discount or incentive for "
            "switching to a one-year or two-year contract."
        )

    # --------------------------------------
    # NEW CUSTOMER STRATEGY
    # --------------------------------------

    if tenure <= 12:

        recommendations.append(
            "Provide onboarding support and regular "
            "check-ins during the customer's early months."
        )

    # --------------------------------------
    # HIGH PRICE STRATEGY
    # --------------------------------------

    if monthly_charges >= 70:

        recommendations.append(
            "Review the customer's pricing plan and "
            "consider offering a loyalty discount or "
            "better-value package."
        )

    # --------------------------------------
    # PAYMENT METHOD STRATEGY
    # --------------------------------------

    if payment_method == "Electronic check":

        recommendations.append(
            "Encourage automatic payment methods by "
            "offering a small billing incentive."
        )

    # --------------------------------------
    # SHAP-BASED RECOMMENDATIONS
    # --------------------------------------

    for factor in risk_factors:

        factor_lower = factor.lower()

        if "tenure" in factor_lower:

            recommendations.append(
                "Focus on early customer engagement "
                "because short tenure may increase churn risk."
            )

        elif "monthlycharges" in factor_lower:

            recommendations.append(
                "Evaluate whether the monthly cost "
                "is creating dissatisfaction."
            )

        elif "contract" in factor_lower:

            recommendations.append(
                "Promote a longer-term contract with "
                "clear financial benefits."
            )

        elif "techsupport" in factor_lower:

            recommendations.append(
                "Offer proactive technical support "
                "to improve customer satisfaction."
            )

        elif "onlinesecurity" in factor_lower:

            recommendations.append(
                "Consider promoting security features "
                "that increase customer value."
            )

    # Remove duplicate recommendations
    recommendations = list(
        dict.fromkeys(recommendations)
    )

    return recommendations


# ==========================================
# CREATE RETENTION REPORT
# ==========================================

def create_retention_report(
    customer,
    churn_probability,
    risk_factors
):

    customer_id = (
        customer["customerID"]
        .iloc[0]
    )

    risk_level = (
        get_risk_level(
            churn_probability
        )
    )

    recommendations = (
        generate_recommendations(
            customer,
            churn_probability,
            risk_factors
        )
    )

    print("\n" + "=" * 70)

    print(
        "AI CUSTOMER RETENTION RECOMMENDATION REPORT"
    )

    print("=" * 70)

    print(
        f"\nCustomer ID: {customer_id}"
    )

    print(
        f"Churn Probability: "
        f"{churn_probability * 100:.2f}%"
    )

    print(
        f"Risk Level: {risk_level}"
    )

    print(
        "\nKEY CHURN RISK FACTORS:"
    )

    for factor in risk_factors:

        print(
            f"• {factor}"
        )

    print(
        "\nRECOMMENDED RETENTION ACTIONS:"
    )

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )

    print("\n" + "=" * 70)

    return recommendations


# ==========================================
# TEST THE RECOMMENDATION ENGINE
# ==========================================

if __name__ == "__main__":

    print(
        "\nTesting AI Recommendation Engine..."
    )

    # Load customer dataset
    df = pd.read_csv(
        "data/processed/featured_customer_data.csv"
    )

    # Example customer
    customer = df[
        df["customerID"] == "7590-VHVEG"
    ]

    # Example churn probability
    churn_probability = 0.8049

    # Example SHAP risk factors
    risk_factors = [

        "tenure",
        "MonthlyCharges",
        "long_term_contract_flag",
        "high_monthly_charge_flag",
        "MultipleLines_No"
    ]

    # Generate report
    create_retention_report(
        customer,
        churn_probability,
        risk_factors
    )