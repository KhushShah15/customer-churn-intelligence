-- =========================================================
-- AI-POWERED RETENTION ANALYSIS
-- =========================================================


-- ---------------------------------------------------------
-- 1. TOP 20 HIGHEST-RISK CUSTOMERS
-- ---------------------------------------------------------

SELECT
    c.customer_id,
    c.contract,
    c.tenure,
    c.internet_service,
    c.payment_method,
    c.monthly_charges,

    ROUND(
        p.churn_probability * 100,
        2
    ) AS churn_probability_pct,

    p.risk_level

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

ORDER BY p.churn_probability DESC

LIMIT 20;


-- ---------------------------------------------------------
-- 2. HIGH-RISK CUSTOMERS
-- ---------------------------------------------------------

SELECT
    c.customer_id,
    c.contract,
    c.tenure,
    c.monthly_charges,
    p.churn_probability,
    p.risk_level

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

WHERE p.risk_level = 'HIGH RISK'

ORDER BY p.churn_probability DESC;


-- ---------------------------------------------------------
-- 3. MONTHLY REVENUE AT HIGH RISK
-- ---------------------------------------------------------

SELECT
    ROUND(
        SUM(c.monthly_charges),
        2
    ) AS monthly_revenue_at_risk

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

WHERE p.risk_level = 'HIGH RISK';


-- ---------------------------------------------------------
-- 4. HIGH-RISK CUSTOMERS BY CONTRACT
-- ---------------------------------------------------------

SELECT
    c.contract,

    COUNT(*) AS high_risk_customers,

    ROUND(
        AVG(p.churn_probability) * 100,
        2
    ) AS average_risk_probability

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

WHERE p.risk_level = 'HIGH RISK'

GROUP BY c.contract

ORDER BY high_risk_customers DESC;


-- ---------------------------------------------------------
-- 5. HIGH-RISK CUSTOMERS BY PAYMENT METHOD
-- ---------------------------------------------------------

SELECT
    c.payment_method,

    COUNT(*) AS high_risk_customers,

    ROUND(
        AVG(p.churn_probability) * 100,
        2
    ) AS average_risk_probability

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

WHERE p.risk_level = 'HIGH RISK'

GROUP BY c.payment_method

ORDER BY high_risk_customers DESC;


-- ---------------------------------------------------------
-- 6. HIGH-RISK REVENUE BY CONTRACT
-- ---------------------------------------------------------

SELECT
    c.contract,

    COUNT(*) AS high_risk_customers,

    ROUND(
        SUM(c.monthly_charges),
        2
    ) AS monthly_revenue_at_risk

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

WHERE p.risk_level = 'HIGH RISK'

GROUP BY c.contract

ORDER BY monthly_revenue_at_risk DESC;


-- ---------------------------------------------------------
-- 7. RETENTION PRIORITY LIST
-- ---------------------------------------------------------

SELECT
    c.customer_id,

    c.contract,

    c.tenure,

    c.monthly_charges,

    ROUND(
        p.churn_probability * 100,
        2
    ) AS churn_probability_pct,

    CASE

        WHEN
            p.churn_probability >= 0.70
            AND c.monthly_charges >= 80

            THEN 'CRITICAL'

        WHEN
            p.churn_probability >= 0.70

            THEN 'HIGH'

        WHEN
            p.churn_probability >= 0.40

            THEN 'MEDIUM'

        ELSE 'LOW'

    END AS retention_priority

FROM customers AS c

INNER JOIN customer_predictions AS p
    ON c.customer_id = p.customer_id

ORDER BY
    p.churn_probability DESC,
    c.monthly_charges DESC;