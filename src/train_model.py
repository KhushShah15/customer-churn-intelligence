import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    REPORTS_PATH
)


# ==========================================
# 1. LOAD FEATURED DATA
# ==========================================

def load_data():

    file_path = (
        PROCESSED_DATA_PATH
        / "featured_customer_data.csv"
    )

    df = pd.read_csv(file_path)

    print("Featured dataset loaded successfully!")

    print(f"Dataset Shape: {df.shape}")

    return df


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df):

    # Target variable
    y = df["Churn"]

    # Features
    X = df.drop(
        columns=[
            "Churn",
            "customerID"
        ]
    )

    print("\nData prepared successfully!")

    print(f"Number of features: {X.shape[1]}")

    return X, y


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("\nTrain/Test Split Completed!")

    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================
# 4. CREATE PREPROCESSOR
# ==========================================

def create_preprocessor(X):

    numerical_features = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # Numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


# ==========================================
# 5. CREATE MODELS
# ==========================================

def get_models():

    models = {

        "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    return models


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    models = get_models()

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 60)

        print(f"TRAINING: {model_name}")

        print("=" * 60)

        # Create pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        # Train model
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        y_pred = pipeline.predict(
            X_test
        )

        # Probability predictions
        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        print("Confusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        # Store results
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            }
        )

        trained_models[
            model_name
        ] = pipeline

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        trained_models
    )


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

def select_best_model(
    results_df,
    trained_models
):

    # Sort using ROC-AUC
    results_df = (
        results_df
        .sort_values(
            by="ROC-AUC",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    best_model_name = (
        results_df
        .iloc[0]["Model"]
    )

    best_model = (
        trained_models[
            best_model_name
        ]
    )

    print("\n" + "=" * 60)

    print("MODEL COMPARISON")

    print("=" * 60)

    print(results_df)

    print(
        f"\nBest Model: "
        f"{best_model_name}"
    )

    print(
        f"Best ROC-AUC: "
        f"{results_df.iloc[0]['ROC-AUC']:.4f}"
    )

    return (
        best_model,
        best_model_name,
        results_df
    )


# ==========================================
# 8. SAVE MODEL
# ==========================================

def save_model(
    model,
    model_name,
    results_df
):

    # Create directories
    MODELS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Model path
    model_path = (
        MODELS_PATH
        / "best_churn_model.joblib"
    )

    # Save model
    joblib.dump(
        model,
        model_path
    )

    # Save model comparison
    results_path = (
        REPORTS_PATH
        / "model_results"
    )

    results_path.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_path = (
        results_path
        / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 60)

    print("MODEL SAVED SUCCESSFULLY!")

    print("=" * 60)

    print(f"Best Model: {model_name}")

    print(
        f"Model Location: "
        f"{model_path}"
    )

    print(
        f"Results Location: "
        f"{comparison_path}"
    )


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # Load data
    df = load_data()

    # Prepare data
    X, y = prepare_data(df)

    # Split data
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X)

    # Train and evaluate models
    (
        results_df,
        trained_models
    ) = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # Select best model
    (
        best_model,
        best_model_name,
        results_df
    ) = select_best_model(
        results_df,
        trained_models
    )

    # Save final model
    save_model(
        best_model,
        best_model_name,
        results_df
    )