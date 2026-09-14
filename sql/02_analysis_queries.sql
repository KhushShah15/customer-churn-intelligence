-- =========================================================
-- CUSTOMER CHURN ANALYSIS
-- =========================================================


-- ---------------------------------------------------------
-- 1. CHURN BY CONTRACT TYPE
-- ---------------------------------------------------------

SELECT
    contract,

    COUNT(*) AS customers,

    SUM(churn) AS churned_customers,

    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY contract

ORDER BY churn_rate DESC;


-- ---------------------------------------------------------
-- 2. CHURN BY INTERNET SERVICE
-- ---------------------------------------------------------

SELECT
    internet_service,

    COUNT(*) AS customers,

    SUM(churn) AS churned_customers,

    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY internet_service

ORDER BY churn_rate DESC;


-- ---------------------------------------------------------
-- 3. CHURN BY PAYMENT METHOD
-- ---------------------------------------------------------

SELECT
    payment_method,

    COUNT(*) AS customers,

    SUM(churn) AS churned_customers,

    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY payment_method

ORDER BY churn_rate DESC;


-- ---------------------------------------------------------
-- 4. CHURN BY SENIOR CITIZEN STATUS
-- ---------------------------------------------------------

SELECT
    senior_citizen,

    COUNT(*) AS customers,

    SUM(churn) AS churned_customers,

    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY senior_citizen

ORDER BY churn_rate DESC;


-- ---------------------------------------------------------
-- 5. AVERAGE TENURE: CHURN VS NON-CHURN
-- ---------------------------------------------------------

SELECT
    churn,

    ROUND(
        AVG(tenure),
        2
    ) AS average_tenure_months

FROM customers

GROUP BY churn;


-- ---------------------------------------------------------
-- 6. AVERAGE MONTHLY CHARGES BY CHURN
-- ---------------------------------------------------------

SELECT
    churn,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS average_monthly_charge

FROM customers

GROUP BY churn;