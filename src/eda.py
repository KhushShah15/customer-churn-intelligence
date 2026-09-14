import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_PATH,
    REPORTS_PATH
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
# CREATE FIGURES DIRECTORY
# ==========================================

def create_output_folder():

    figures_path = REPORTS_PATH / "figures"

    figures_path.mkdir(
        parents=True,
        exist_ok=True
    )

    return figures_path


# ==========================================
# CHART 1: CHURN DISTRIBUTION
# ==========================================

def churn_distribution(df, figures_path):

    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(8, 5))

    churn_counts.plot(
        kind="bar"
    )

    plt.title("Customer Churn Distribution")

    plt.xlabel("Churn")

    plt.ylabel("Number of Customers")

    plt.xticks(
        ticks=[0, 1],
        labels=["No Churn", "Churn"],
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        figures_path / "01_churn_distribution.png"
    )

    plt.close()


# ==========================================
# CHART 2: CHURN BY CONTRACT
# ==========================================

def churn_by_contract(df, figures_path):

    churn_rate = (
        df.groupby("Contract")["Churn"]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    plt.figure(figsize=(8, 5))

    churn_rate.plot(
        kind="bar"
    )

    plt.title("Churn Rate by Contract Type")

    plt.xlabel("Contract Type")

    plt.ylabel("Churn Rate (%)")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        figures_path / "02_churn_by_contract.png"
    )

    plt.close()


# ==========================================
# CHART 3: CHURN BY INTERNET SERVICE
# ==========================================

def churn_by_internet_service(df, figures_path):

    churn_rate = (
        df.groupby("InternetService")["Churn"]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    plt.figure(figsize=(8, 5))

    churn_rate.plot(
        kind="bar"
    )

    plt.title("Churn Rate by Internet Service")

    plt.xlabel("Internet Service")

    plt.ylabel("Churn Rate (%)")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        figures_path / "03_churn_by_internet_service.png"
    )

    plt.close()


# ==========================================
# CHART 4: CHURN BY PAYMENT METHOD
# ==========================================

def churn_by_payment_method(df, figures_path):

    churn_rate = (
        df.groupby("PaymentMethod")["Churn"]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    plt.figure(figsize=(10, 6))

    churn_rate.plot(
        kind="bar"
    )

    plt.title("Churn Rate by Payment Method")

    plt.xlabel("Payment Method")

    plt.ylabel("Churn Rate (%)")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "04_churn_by_payment_method.png"
    )

    plt.close()


# ==========================================
# CHART 5: TENURE DISTRIBUTION
# ==========================================

def tenure_distribution(df, figures_path):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["tenure"],
        bins=20
    )

    plt.title("Customer Tenure Distribution")

    plt.xlabel("Tenure (Months)")

    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "05_tenure_distribution.png"
    )

    plt.close()


# ==========================================
# CHART 6: MONTHLY CHARGES VS CHURN
# ==========================================

def monthly_charges_by_churn(df, figures_path):

    no_churn = df[
        df["Churn"] == 0
    ]["MonthlyCharges"]

    churn = df[
        df["Churn"] == 1
    ]["MonthlyCharges"]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [no_churn, churn],
        tick_labels=[
            "No Churn",
            "Churn"
        ]
    )

    plt.title(
        "Monthly Charges by Churn Status"
    )

    plt.ylabel("Monthly Charges")

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "06_monthly_charges_vs_churn.png"
    )

    plt.close()


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

def print_business_insights(df):

    print("\n" + "=" * 60)

    print("BUSINESS INSIGHTS")

    print("=" * 60)

    churn_rate = (
        df["Churn"].mean()
        * 100
    )

    print(
        f"\nOverall Churn Rate: "
        f"{churn_rate:.2f}%"
    )

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .mean()
        * 100
    )

    highest_contract = (
        contract_churn.idxmax()
    )

    print(
        f"\nHighest Churn Contract: "
        f"{highest_contract}"
    )

    print(
        f"Churn Rate: "
        f"{contract_churn.max():.2f}%"
    )

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .mean()
        * 100
    )

    highest_payment = (
        payment_churn.idxmax()
    )

    print(
        f"\nHighest Risk Payment Method: "
        f"{highest_payment}"
    )

    print(
        f"Churn Rate: "
        f"{payment_churn.max():.2f}%"
    )

    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .mean()
        * 100
    )

    highest_internet = (
        internet_churn.idxmax()
    )

    print(
        f"\nHighest Risk Internet Service: "
        f"{highest_internet}"
    )

    print(
        f"Churn Rate: "
        f"{internet_churn.max():.2f}%"
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # Load data
    df = load_data()

    # Create figures folder
    figures_path = create_output_folder()

    # Create charts
    churn_distribution(
        df,
        figures_path
    )

    churn_by_contract(
        df,
        figures_path
    )

    churn_by_internet_service(
        df,
        figures_path
    )

    churn_by_payment_method(
        df,
        figures_path
    )

    tenure_distribution(
        df,
        figures_path
    )

    monthly_charges_by_churn(
        df,
        figures_path
    )

    # Print insights
    print_business_insights(df)

    print("\nEDA completed successfully!")

    print(
        f"Charts saved in: "
        f"{figures_path}"
    )