import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b"
)


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please check the .env file."
    )


client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# HELPER: CONVERT RISK FACTORS TO TEXT
# ============================================================

def format_risk_factors(risk_factors):

    if not risk_factors:
        return "No specific SHAP risk factors available."

    formatted_factors = []

    for factor in risk_factors:

        # If factor is a dictionary
        if isinstance(factor, dict):

            feature = factor.get(
                "feature",
                factor.get("name", "Unknown factor")
            )

            value = factor.get(
                "shap_value",
                factor.get("value", "")
            )

            if value != "":
                formatted_factors.append(
                    f"- {feature}: SHAP impact {value}"
                )
            else:
                formatted_factors.append(
                    f"- {feature}"
                )

        # If factor is a tuple/list
        elif isinstance(factor, (tuple, list)):

            if len(factor) >= 2:
                formatted_factors.append(
                    f"- {factor[0]}: SHAP impact {factor[1]}"
                )
            else:
                formatted_factors.append(
                    f"- {factor[0]}"
                )

        # If factor is already text
        else:
            formatted_factors.append(
                f"- {factor}"
            )

    return "\n".join(
        formatted_factors
    )


# ============================================================
# HELPER: FORMAT RULE-BASED RECOMMENDATIONS
# ============================================================

def format_rule_recommendations(
    rule_recommendations
):

    if not rule_recommendations:
        return "No rule-based recommendations available."

    return "\n".join(
        [
            f"- {recommendation}"
            for recommendation
            in rule_recommendations
        ]
    )


# ============================================================
# GENERATE PERSONALIZED LLM STRATEGY
# ============================================================

def generate_llm_strategy(
    customer,
    churn_probability,
    risk_level,
    risk_factors,
    rule_recommendations
):

    # --------------------------------------------------------
    # READ CUSTOMER DETAILS
    # --------------------------------------------------------

    if hasattr(customer, "iloc"):

        if hasattr(customer, "columns"):
            row = customer.iloc[0]
        else:
            row = customer

    else:
        row = customer


    customer_id = row.get(
        "customerID",
        "Unknown"
    )

    tenure = row.get(
        "tenure",
        "Unknown"
    )

    contract = row.get(
        "Contract",
        "Unknown"
    )

    monthly_charges = row.get(
        "MonthlyCharges",
        "Unknown"
    )

    payment_method = row.get(
        "PaymentMethod",
        "Unknown"
    )

    internet_service = row.get(
        "InternetService",
        "Unknown"
    )


    # --------------------------------------------------------
    # FORMAT INPUT INFORMATION
    # --------------------------------------------------------

    shap_text = format_risk_factors(
        risk_factors
    )

    rules_text = format_rule_recommendations(
        rule_recommendations
    )


    # --------------------------------------------------------
    # CREATE LLM PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a senior telecom customer-retention strategist.

Your job is to turn machine-learning predictions,
SHAP explanations, and existing business rules into a
clear personalized customer-retention strategy.

CUSTOMER PROFILE
----------------
Customer ID: {customer_id}
Churn Probability: {churn_probability * 100:.2f}%
Risk Level: {risk_level}
Tenure: {tenure} months
Contract: {contract}
Monthly Charges: {monthly_charges}
Payment Method: {payment_method}
Internet Service: {internet_service}


MODEL-IDENTIFIED SHAP FACTORS
-----------------------------
{shap_text}


EXISTING RULE-BASED RECOMMENDATIONS
-----------------------------------
{rules_text}


IMPORTANT RULES
---------------

Use ONLY the customer data, model prediction,
SHAP factors, and rule-based recommendations
provided above.

Never invent or assume information that is not
explicitly present in the supplied data.

Do NOT infer:
- customer dissatisfaction
- complaints
- service problems
- payment friction
- financial difficulties
- competitor activity
- usage decline
- customer preferences
- support history
- engagement level
- account setup problems
- reasons why the customer selected a payment method
- reasons why the customer selected a service

SHAP factors represent MODEL INFLUENCE only.

A positive SHAP value means that the feature pushed
the model prediction toward higher churn risk.

A negative SHAP value means that the feature pushed
the model prediction toward lower churn risk.

Do NOT describe SHAP factors as proven causes of churn.

For example:

BAD:
"Electronic check creates payment friction."

GOOD:
"Electronic check was identified by the model as
a factor associated with this customer's churn prediction."

BAD:
"The customer is dissatisfied with monthly charges."

GOOD:
"Monthly Charges contributed to the model's
higher churn-risk prediction."

Recommendations must be based only on the supplied
customer information and existing business rules.

When evidence is insufficient, explicitly avoid making
a claim rather than guessing.

OUTPUT COMPLETENESS RULES
-------------------------

Complete every section fully.

Never leave a heading, bullet, Action, Rationale,
Risk Driver, Offer Strategy, or Communication Approach blank.

If there is not enough evidence for a statement,
write:

"Not enough evidence available from the supplied data."

Do not stop the response in the middle of a section.

Every recommended action must include both:

- Action
- Rationale

The rationale must explain which supplied customer field,
SHAP factor, prediction, or business rule supports the action.

For example, instead of:

C. Early Engagement & Onboarding Support

• Action: Provide onboarding support...
• Rationale:

we want:

C. Early Engagement & Onboarding Support

• Action: Provide onboarding support and regular check-ins.

• Rationale: The customer has only 1 month of tenure, and
tenure is one of the strongest model-identified factors
increasing the churn prediction.

"""

    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        response = (
            client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,

                reasoning_effort="none",

                reasoning_format="hidden",

                max_completion_tokens=1600
            )
        )


        strategy = (
            response
            .choices[0]
            .message
            .content
        )


        if not strategy:
            raise ValueError(
                "Groq returned an empty response."
            )


        return strategy.strip()


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    except Exception as error:

        print(
            f"\nGroq API Error: {error}"
        )

        fallback = (
            "\n".join(
                [
                    f"{index}. {recommendation}"
                    for index, recommendation
                    in enumerate(
                        rule_recommendations,
                        start=1
                    )
                ]
            )
        )

        return (
            "The generative AI service is currently "
            "unavailable.\n\n"
            "Rule-Based Retention Recommendations:\n"
            + fallback
        )