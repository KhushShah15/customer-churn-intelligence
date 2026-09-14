import pandas as pd
from src.llm_recommendations import generate_llm_strategy


from src.predict import (
    load_model,
    load_customer_data,
    get_customer_data,
    prepare_customer_features,
    predict_churn,
    get_risk_level
)

from src.shap_explainer import (
    prepare_features,
    explain_customer,
    get_top_factors
)

from src.ai_recommendations import (
    generate_recommendations
)


# ==========================================
# CREATE COMPLETE CHURN INTELLIGENCE REPORT
# ==========================================

def create_intelligence_report(
    customer_id,
    customer,
    probability,
    prediction,
    risk_increasing,
    risk_reducing,
    recommendations
):

    risk_level = get_risk_level(
        probability
    )

    print("\n")
    print("=" * 70)
    print("CUSTOMER CHURN INTELLIGENCE REPORT")
    print("=" * 70)

    print(f"\nCustomer ID: {customer_id}")

    print(
        f"Churn Probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk Level: {risk_level}"
    )

    print(
        f"Prediction: "
        f"{'Likely to Churn' if prediction == 1 else 'Likely to Stay'}"
    )

    print("\n" + "-" * 70)
    print("CUSTOMER PROFILE")
    print("-" * 70)

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

    print("\n" + "-" * 70)
    print("TOP FACTORS INCREASING CHURN RISK")
    print("-" * 70)

    for _, row in risk_increasing.iterrows():

        print(
            f"• {row['Feature']} "
            f"(Impact: {row['SHAP_Value']:.4f})"
        )

    print("\n" + "-" * 70)
    print("TOP FACTORS REDUCING CHURN RISK")
    print("-" * 70)

    for _, row in risk_reducing.iterrows():

        print(
            f"• {row['Feature']} "
            f"(Impact: {row['SHAP_Value']:.4f})"
        )

    print("\n" + "-" * 70)
    print("RULE-BASED RETENTION RECOMMENDATIONS")
    print("-" * 70)

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )

    print("\n" + "=" * 70)

    print(
        "END OF CUSTOMER INTELLIGENCE REPORT"
    )

    print("=" * 70)


# ==========================================
# MAIN INTELLIGENCE PIPELINE
# ==========================================

def run_churn_intelligence(
    customer_id
):

    print("\nLoading trained model...")

    model = load_model()

    print("\nLoading customer dataset...")

    df = load_customer_data()

    # --------------------------------------
    # FIND CUSTOMER
    # --------------------------------------

    customer = get_customer_data(
        df,
        customer_id
    )

    if customer is None:

        return

    # --------------------------------------
    # PREDICT CHURN
    # --------------------------------------

    customer_features = (
        prepare_customer_features(
            customer
        )
    )

    prediction, probability = (
        predict_churn(
            model,
            customer_features
        )
    )

    # --------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------

    print(
        "\nRunning SHAP explainability analysis..."
    )

    X_background = prepare_features(
        df
    )

    shap_values, feature_names = (
        explain_customer(
            model,
            X_background,
            customer_features
        )
    )

    risk_increasing, risk_reducing = (
        get_top_factors(
            shap_values,
            feature_names
        )
    )

    # --------------------------------------
    # EXTRACT RISK FACTOR NAMES
    # --------------------------------------

    risk_factors = (
        risk_increasing[
            "Feature"
        ]
        .tolist()
    )

    # --------------------------------------
    # GENERATE AI RECOMMENDATIONS
    # --------------------------------------

    recommendations = (
        generate_recommendations(
            customer,
            probability,
            risk_factors
        )
    )
    
    # ============================================================
    # GENERATE GROQ LLM RETENTION STRATEGY
    # ============================================================

    llm_strategy = generate_llm_strategy(
    customer=customer,
    churn_probability=probability,
    risk_level=get_risk_level(probability),
    risk_factors=risk_factors,
    rule_recommendations=recommendations
    )

    # --------------------------------------
    # DISPLAY FINAL REPORT
    # --------------------------------------

    create_intelligence_report(
        customer_id,
        customer,
        probability,
        prediction,
        risk_increasing,
        risk_reducing,
        recommendations
    )
    
    print("\n" + "=" * 70)
    print("GROQ AI PERSONALIZED RETENTION STRATEGY")
    print("=" * 70)
    
    print("\n" + llm_strategy)
    
    print("\n" + "=" * 70)


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    print(
        "\nCUSTOMER CHURN INTELLIGENCE SYSTEM"
    )

    print(
        "\nExample Customer IDs:"
    )

    df = load_customer_data()

    print(
        df["customerID"]
        .head()
        .tolist()
    )

    customer_id = input(
        "\nEnter Customer ID: "
    ).strip()

    run_churn_intelligence(
        customer_id
    )