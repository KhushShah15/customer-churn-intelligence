import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    REPORTS_PATH
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = (
    PROCESSED_DATA_PATH
    / "featured_customer_data.csv"
)

MODEL_PATH = (
    MODELS_PATH
    / "best_churn_model.joblib"
)

OUTPUT_PATH = (
    REPORTS_PATH
    / "powerbi"
    / "model_test_predictions.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading featured dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset loaded successfully: {df.shape}"
    )

    return df


# ==========================================
# RECREATE TEST SET
# ==========================================

def create_test_set(df):

    # Save customer IDs separately
    customer_ids = df["customerID"]

    # Target
    y = df["Churn"]

    # ML features
    X = df.drop(
        columns=[
            "customerID",
            "Churn"
        ]
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test
    ) = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTest samples: {len(X_test)}"
    )

    return (
        X_test,
        y_test,
        id_test
    )


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    print("\nLoading saved churn model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully!"
    )

    return model


# ==========================================
# CREATE TEST PREDICTIONS
# ==========================================

def generate_predictions(
    model,
    X_test,
    y_test,
    id_test
):

    print(
        "\nGenerating test-set predictions..."
    )

    predicted = model.predict(
        X_test
    )

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    results = pd.DataFrame(
        {
            "customer_id":
                id_test.values,

            "actual_churn":
                y_test.values,

            "predicted_churn":
                predicted,

            "churn_probability":
                probability
        }
    )

    results[
        "actual_label"
    ] = results[
        "actual_churn"
    ].map(
        {
            0: "Stay",
            1: "Churn"
        }
    )

    results[
        "predicted_label"
    ] = results[
        "predicted_churn"
    ].map(
        {
            0: "Stay",
            1: "Churn"
        }
    )

    print(
        "Test predictions generated!"
    )

    return results


# ==========================================
# SAVE FILE
# ==========================================

def save_results(results):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nPower BI evaluation file saved!"
    )

    print(
        f"Location: {OUTPUT_PATH}"
    )


# ==========================================
# VERIFY CONFUSION MATRIX
# ==========================================

def verify_results(results):

    confusion = pd.crosstab(
        results["actual_label"],
        results["predicted_label"]
    )

    print("\n" + "=" * 60)
    print("TEST-SET CONFUSION MATRIX")
    print("=" * 60)

    print(confusion)

    print(
        f"\nTotal Test Rows: "
        f"{len(results)}"
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    df = load_data()

    (
        X_test,
        y_test,
        id_test
    ) = create_test_set(df)

    model = load_model()

    results = generate_predictions(
        model,
        X_test,
        y_test,
        id_test
    )

    save_results(
        results
    )

    verify_results(
        results
    )

    print("\n" + "=" * 60)
    print(
        "MODEL EVALUATION EXPORT COMPLETED!"
    )
    print("=" * 60)