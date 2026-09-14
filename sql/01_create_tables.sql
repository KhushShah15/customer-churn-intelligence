-- =========================================================
-- CUSTOMER CHURN INTELLIGENCE SYSTEM
-- BUSINESS KPI ANALYSIS
-- =========================================================


-- ---------------------------------------------------------
-- 1. TOTAL CUSTOMERS
-- ---------------------------------------------------------

SELECT
    COUNT(*) AS total_customers
FROM customers;


-- ---------------------------------------------------------
-- 2. ACTUAL CHURNED CUSTOMERS
-- ---------------------------------------------------------

SELECT
    COUNT(*) AS churned_customers
FROM customers
WHERE churn = 1;


-- ---------------------------------------------------------
-- 3. ACTUAL CHURN RATE
-- ---------------------------------------------------------

SELECT
    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS actual_churn_rate
FROM customers;


-- ---------------------------------------------------------
-- 4. AVERAGE MONTHLY REVENUE
-- ---------------------------------------------------------

SELECT
    ROUND(
        AVG(monthly_charges),
        2
    ) AS average_monthly_charge
FROM customers;


-- ---------------------------------------------------------
-- 5. TOTAL MONTHLY REVENUE
-- ---------------------------------------------------------

SELECT
    ROUND(
        SUM(monthly_charges),
        2
    ) AS total_monthly_revenue
FROM customers;


-- ---------------------------------------------------------
-- 6. AVERAGE PREDICTED CHURN PROBABILITY
-- ---------------------------------------------------------

SELECT
    ROUND(
        AVG(churn_probability) * 100,
        2
    ) AS average_churn_probability
FROM customer_predictions;


-- ---------------------------------------------------------
-- 7. CUSTOMER RISK DISTRIBUTION
-- ---------------------------------------------------------

SELECT
    risk_level,
    COUNT(*) AS customers,

    ROUND(
        COUNT(*) * 100.0 /
        (
            SELECT COUNT(*)
            FROM customer_predictions
        ),
        2
    ) AS percentage

FROM customer_predictions

GROUP BY risk_level

ORDER BY customers DESC;