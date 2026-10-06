from .schema import Column, DTypes

# ==============================================================================
# 1. TELECOM DOMAIN PRESET
# ==============================================================================
TELECOM_PRESET = {
    "name": "Telecom",
    "description": "Comprehensive telecom subscriber churn dataset with demographics, contract terms, service subscriptions, and usage charges.",
    "target_churn_rate": 0.265,
    "noise_level": 0.3,
    "columns": [
        Column("customer_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Account & Demographics"),
        Column("gender", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Male", "Female"]}, 0.0, category="Account & Demographics"),
        Column("senior_citizen", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.16}, 0.2, category="Account & Demographics"),
        Column("partner", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.48}, -0.2, category="Account & Demographics"),
        Column("dependents", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.30}, -0.2, category="Account & Demographics"),
        Column("state", DTypes.CATEGORICAL, "faker.state", {}, 0.0, category="Account & Demographics"),
        Column("city", DTypes.CATEGORICAL, "faker.city", {}, 0.0, category="Account & Demographics"),
        
        Column("tenure_months", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 72}, -0.85, category="Contract & Billing", min_val=1.0, max_val=72.0),
        Column("contract_type", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Month-to-month", "One year", "Two year"]}, -1.2, category="Contract & Billing"),
        Column("paperless_billing", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.59}, 0.25, category="Contract & Billing"),
        Column("payment_method", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]}, 0.2, category="Contract & Billing"),
        Column("billing_cycle", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Monthly", "Quarterly", "Annual"]}, -0.3, category="Contract & Billing"),
        Column("auto_renew", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.42}, -0.4, category="Contract & Billing"),

        Column("phone_service", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.90}, 0.0, category="Phone & Voice Services"),
        Column("multiple_lines", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["No", "Yes", "No phone service"]}, 0.1, category="Phone & Voice Services"),
        Column("international_plan", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.15}, 0.35, category="Phone & Voice Services"),
        Column("voicemail_plan", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.28}, -0.2, category="Phone & Voice Services"),

        Column("internet_service", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["DSL", "Fiber optic", "No"]}, 0.45, category="Internet & Entertainment"),
        Column("online_security", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, -0.35, category="Internet & Entertainment"),
        Column("online_backup", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, -0.25, category="Internet & Entertainment"),
        Column("device_protection", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, -0.2, category="Internet & Entertainment"),
        Column("tech_support", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, -0.45, category="Internet & Entertainment"),
        Column("streaming_tv", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, 0.15, category="Internet & Entertainment"),
        Column("streaming_movies", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Yes", "No", "No internet service"]}, 0.15, category="Internet & Entertainment"),
        Column("unlimited_data", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.65}, -0.15, category="Internet & Entertainment"),

        Column("monthly_charges", DTypes.NUMERIC, "numpy.normal", {"loc": 64.76, "scale": 30.09}, 0.55, category="Usage & Charges", min_val=18.0),
        Column("total_charges", DTypes.NUMERIC, "numpy.normal", {"loc": 2283.3, "scale": 2266.77}, -0.1, category="Usage & Charges", min_val=18.0),
        Column("extra_data_charges", DTypes.NUMERIC, "numpy.poisson", {"lam": 10}, 0.4, category="Usage & Charges", min_val=0.0),
        Column("avg_monthly_gb_download", DTypes.NUMERIC, "numpy.normal", {"loc": 25.0, "scale": 15.0}, -0.2, category="Usage & Charges", min_val=0.0),
        Column("customer_service_calls", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.5}, 0.7, category="Usage & Charges", min_val=0.0),
        Column("satisfaction_score", DTypes.NUMERIC, "numpy.uniform", {"low": 1.0, "high": 5.0}, -0.8, category="Usage & Charges", min_val=1.0, max_val=5.0),
        Column("clv_estimate", DTypes.NUMERIC, "numpy.normal", {"loc": 4500, "scale": 1800}, -0.3, category="Usage & Charges", min_val=500.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["tenure_months", "total_charges"], "matrix": [[1.0, 0.85], [0.85, 1.0]]},
        {"type": "correlation", "columns": ["monthly_charges", "extra_data_charges"], "matrix": [[1.0, 0.40], [0.40, 1.0]]},
        {"type": "conditional", "target": "multiple_lines", "condition_col": "phone_service", "condition_val": False, "dist": "choice", "params": {"elements": ["No phone service"]}},
        {"type": "conditional", "target": "online_security", "condition_col": "internet_service", "condition_val": "No", "dist": "choice", "params": {"elements": ["No internet service"]}},
        {"type": "conditional", "target": "tech_support", "condition_col": "internet_service", "condition_val": "No", "dist": "choice", "params": {"elements": ["No internet service"]}},
        {"type": "interaction", "conditions": [{"col": "contract_type", "op": "==", "val": "Month-to-month"}, {"col": "tech_support", "op": "==", "val": "No"}], "churn_boost": 0.85},
        {"type": "interaction", "conditions": [{"col": "internet_service", "op": "==", "val": "Fiber optic"}, {"col": "online_security", "op": "==", "val": "No"}], "churn_boost": 0.65},
        {"type": "interaction", "conditions": [{"col": "customer_service_calls", "op": ">=", "val": "4"}, {"col": "satisfaction_score", "op": "<=", "val": "2"}], "churn_boost": 1.1},
        {"type": "nonlinear", "column": "tenure_months", "transform": "step", "params": {"threshold": 12, "below": 0.7, "above": -0.3}}
    ]
}

# ==============================================================================
# 2. SAAS DOMAIN PRESET
# ==============================================================================
SAAS_PRESET = {
    "name": "SaaS",
    "description": "B2B Subscription & SaaS churn dataset with organizational tiers, user engagement, API load, and CSM relationship health.",
    "target_churn_rate": 0.22,
    "noise_level": 0.3,
    "columns": [
        Column("user_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Company Profile"),
        Column("company_size", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["startup", "smb", "mid_market", "enterprise"]}, -0.4, category="Company Profile"),
        Column("industry", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Tech", "Finance", "Healthcare", "Retail", "Education", "Manufacturing"]}, 0.0, category="Company Profile"),
        Column("country", DTypes.CATEGORICAL, "faker.country", {}, 0.0, category="Company Profile"),
        Column("employee_count", DTypes.NUMERIC, "numpy.lognormal", {"mean": 4.5, "sigma": 1.5}, -0.2, category="Company Profile", min_val=1.0),
        Column("annual_revenue_bracket", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["<$1M", "$1M-$10M", "$10M-$50M", ">$50M"]}, -0.3, category="Company Profile"),

        Column("plan_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["free", "basic", "pro", "enterprise"]}, -0.65, category="Subscription & Plan"),
        Column("billing_frequency", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["monthly", "annual", "multi_year"]}, -0.5, category="Subscription & Plan"),
        Column("seat_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 25}, -0.3, category="Subscription & Plan", min_val=1.0),
        Column("active_seats_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.1, "high": 1.0}, -0.7, category="Subscription & Plan", min_val=0.0, max_val=1.0),
        Column("signup_days_ago", DTypes.NUMERIC, "numpy.uniform", {"low": 30, "high": 1800}, -0.25, category="Subscription & Plan", min_val=1.0),
        Column("contract_months", DTypes.NUMERIC, "numpy.random.choice", {"a": [1, 12, 24, 36], "p": [0.4, 0.4, 0.15, 0.05]}, -0.6, category="Subscription & Plan", min_val=1.0),
        Column("auto_renew", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.72}, -0.45, category="Subscription & Plan"),

        Column("logins_last_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 18}, -0.8, category="Product Engagement", min_val=0.0),
        Column("features_used_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.05, "high": 1.0}, -0.75, category="Product Engagement", min_val=0.0, max_val=1.0),
        Column("api_calls_last_30d", DTypes.NUMERIC, "numpy.lognormal", {"mean": 8, "sigma": 2}, -0.3, category="Product Engagement", min_val=0.0),
        Column("storage_used_gb", DTypes.NUMERIC, "numpy.lognormal", {"mean": 3, "sigma": 1.2}, -0.2, category="Product Engagement", min_val=0.0),
        Column("dashboard_views_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 45}, -0.4, category="Product Engagement", min_val=0.0),
        Column("reports_exported_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 6}, -0.3, category="Product Engagement", min_val=0.0),
        Column("integrations_connected", DTypes.NUMERIC, "numpy.poisson", {"lam": 4}, -0.5, category="Product Engagement", min_val=0.0),
        Column("mobile_app_user", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.38}, -0.2, category="Product Engagement"),

        Column("support_tickets_90d", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, 0.5, category="Customer Success & Feedback", min_val=0.0),
        Column("avg_resolution_hours", DTypes.NUMERIC, "numpy.normal", {"loc": 14, "scale": 8}, 0.3, category="Customer Success & Feedback", min_val=1.0),
        Column("csat_score", DTypes.NUMERIC, "numpy.uniform", {"low": 1.0, "high": 5.0}, -0.7, category="Customer Success & Feedback", min_val=1.0, max_val=5.0),
        Column("nps_score", DTypes.NUMERIC, "numpy.uniform", {"low": 0, "high": 10}, -0.9, category="Customer Success & Feedback", min_val=0.0, max_val=10.0),
        Column("has_dedicated_csm", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.18}, -0.65, category="Customer Success & Feedback"),
        Column("onboarding_completed", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.82}, -0.55, category="Customer Success & Feedback"),

        Column("monthly_recurring_revenue", DTypes.NUMERIC, "numpy.lognormal", {"mean": 6.2, "sigma": 1.1}, -0.15, category="Financial Metrics", min_val=20.0),
        Column("total_contract_value", DTypes.NUMERIC, "numpy.lognormal", {"mean": 8.5, "sigma": 1.4}, -0.2, category="Financial Metrics", min_val=100.0),
        Column("lifetime_value", DTypes.NUMERIC, "numpy.normal", {"loc": 12000, "scale": 6000}, -0.3, category="Financial Metrics", min_val=500.0),
        Column("payment_failures_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.4}, 0.8, category="Financial Metrics", min_val=0.0)
    ],
    "relationships": [
        {"type": "conditional", "target": "has_dedicated_csm", "condition_col": "company_size", "condition_val": "enterprise", "dist": "choice", "params": {"elements": [True]}},
        {"type": "correlation", "columns": ["features_used_pct", "logins_last_30d"], "matrix": [[1.0, 0.65], [0.65, 1.0]]},
        {"type": "correlation", "columns": ["seat_count", "monthly_recurring_revenue"], "matrix": [[1.0, 0.70], [0.70, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "plan_tier", "op": "==", "val": "free"}, {"col": "support_tickets_90d", "op": ">=", "val": "3"}], "churn_boost": 0.95},
        {"type": "interaction", "conditions": [{"col": "logins_last_30d", "op": "<", "val": "5"}, {"col": "features_used_pct", "op": "<", "val": "0.25"}], "churn_boost": 1.25},
        {"type": "interaction", "conditions": [{"col": "payment_failures_count", "op": ">=", "val": "2"}], "churn_boost": 0.8},
        {"type": "nonlinear", "column": "nps_score", "transform": "step", "params": {"threshold": 6, "below": 1.0, "above": -0.5}}
    ]
}

# ==============================================================================
# 3. BANKING DOMAIN PRESET
# ==============================================================================
BANKING_PRESET = {
    "name": "Banking",
    "description": "Retail and commercial banking customer churn with demographic portfolios, credit history, deposit volumes, and digital channel engagement.",
    "target_churn_rate": 0.20,
    "noise_level": 0.3,
    "columns": [
        Column("customer_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Customer Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 38.9, "scale": 10.5}, 0.25, category="Customer Profile", min_val=18.0, max_val=95.0),
        Column("gender", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Male", "Female"]}, -0.1, category="Customer Profile"),
        Column("marital_status", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Single", "Married", "Divorced"]}, 0.0, category="Customer Profile"),
        Column("occupation", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Professional", "Salaried", "Self-employed", "Retired", "Student"]}, -0.15, category="Customer Profile"),
        Column("geography", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["France", "Germany", "Spain"]}, 0.0, category="Customer Profile"),
        Column("household_income", DTypes.NUMERIC, "numpy.normal", {"loc": 65000, "scale": 32000}, -0.2, category="Customer Profile", min_val=12000.0),

        Column("account_type", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Checking", "Savings", "Premium", "Business"]}, -0.2, category="Account Details"),
        Column("account_age_months", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 180}, -0.25, category="Account Details", min_val=1.0),
        Column("num_products", DTypes.NUMERIC, "numpy.random.choice", {"a": [1, 2, 3, 4], "p": [0.5, 0.44, 0.05, 0.01]}, -0.5, category="Account Details", min_val=1.0, max_val=4.0),
        Column("has_credit_card", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.70}, -0.1, category="Account Details"),
        Column("card_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Standard", "Gold", "Platinum", "None"]}, -0.3, category="Account Details"),
        Column("is_active_member", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.51}, -0.85, category="Account Details"),
        Column("direct_deposit_active", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.62}, -0.6, category="Account Details"),

        Column("credit_score", DTypes.NUMERIC, "numpy.normal", {"loc": 650.5, "scale": 96.6}, -0.15, category="Balances & Wealth", min_val=300.0, max_val=850.0),
        Column("balance", DTypes.NUMERIC, "numpy.normal", {"loc": 76485.8, "scale": 62397.4}, 0.2, category="Balances & Wealth", min_val=0.0),
        Column("savings_balance", DTypes.NUMERIC, "numpy.normal", {"loc": 32000, "scale": 28000}, -0.3, category="Balances & Wealth", min_val=0.0),
        Column("investment_balance", DTypes.NUMERIC, "numpy.normal", {"loc": 15000, "scale": 25000}, -0.35, category="Balances & Wealth", min_val=0.0),
        Column("estimated_salary", DTypes.NUMERIC, "numpy.uniform", {"low": 12000, "high": 200000}, 0.0, category="Balances & Wealth", min_val=1000.0),
        Column("debt_to_income_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.05, "high": 0.65}, 0.45, category="Balances & Wealth", min_val=0.0, max_val=1.0),
        Column("has_mortgage", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.35}, -0.4, category="Balances & Wealth"),

        Column("num_transactions_90d", DTypes.NUMERIC, "numpy.poisson", {"lam": 35}, -0.6, category="Transactions & Digital Activity", min_val=0.0),
        Column("avg_transaction_amount", DTypes.NUMERIC, "numpy.normal", {"loc": 85, "scale": 45}, -0.1, category="Transactions & Digital Activity", min_val=2.0),
        Column("mobile_app_logins_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 22}, -0.5, category="Transactions & Digital Activity", min_val=0.0),
        Column("atm_visits_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, -0.1, category="Transactions & Digital Activity", min_val=0.0),
        Column("overdraft_events_year", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.5}, 0.7, category="Transactions & Digital Activity", min_val=0.0),
        Column("customer_complaints_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.3}, 0.9, category="Transactions & Digital Activity", min_val=0.0),
        Column("branch_visits_last_year", DTypes.NUMERIC, "numpy.poisson", {"lam": 4}, 0.1, category="Transactions & Digital Activity", min_val=0.0)
    ],
    "relationships": [
        {"type": "conditional", "target": "balance", "condition_col": "geography", "condition_val": "Germany", "dist": "normal", "params": {"loc": 119730, "scale": 27000}},
        {"type": "correlation", "columns": ["age", "account_age_months"], "matrix": [[1.0, 0.35], [0.35, 1.0]]},
        {"type": "correlation", "columns": ["balance", "savings_balance"], "matrix": [[1.0, 0.50], [0.50, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "geography", "op": "==", "val": "Germany"}, {"col": "gender", "op": "==", "val": "Female"}], "churn_boost": 0.5},
        {"type": "interaction", "conditions": [{"col": "num_products", "op": "==", "val": 1}, {"col": "is_active_member", "op": "==", "val": False}], "churn_boost": 0.85},
        {"type": "interaction", "conditions": [{"col": "overdraft_events_year", "op": ">=", "val": "2"}, {"col": "customer_complaints_count", "op": ">=", "val": "1"}], "churn_boost": 1.1},
        {"type": "nonlinear", "column": "age", "transform": "u_curve", "params": {"center": 50, "scale": 0.002}}
    ]
}

# ==============================================================================
# 4. E-COMMERCE DOMAIN PRESET
# ==============================================================================
ECOMMERCE_PRESET = {
    "name": "E-Commerce",
    "description": "Online retail customer lifecycle dataset including order frequency, cart friction, marketing channel responses, and product category interests.",
    "target_churn_rate": 0.24,
    "noise_level": 0.3,
    "columns": [
        Column("customer_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Customer Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 34.5, "scale": 11.5}, 0.1, category="Customer Profile", min_val=16.0, max_val=90.0),
        Column("gender", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Male", "Female", "Other"]}, 0.0, category="Customer Profile"),
        Column("location_region", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["North America", "Europe", "Asia-Pacific", "Latin America"]}, 0.0, category="Customer Profile"),
        Column("income_bracket", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Low", "Medium", "High", "Affluent"]}, -0.2, category="Customer Profile"),
        Column("member_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Standard", "Bronze", "Silver", "Gold", "VIP"]}, -0.5, category="Customer Profile"),

        Column("days_since_first_order", DTypes.NUMERIC, "numpy.uniform", {"low": 30, "high": 1800}, -0.2, category="Purchase History", min_val=1.0),
        Column("days_since_last_order", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 365}, 0.95, category="Purchase History", min_val=0.0),
        Column("order_count", DTypes.NUMERIC, "numpy.lognormal", {"mean": 2, "sigma": 1}, -0.75, category="Purchase History", min_val=1.0),
        Column("avg_order_value", DTypes.NUMERIC, "numpy.normal", {"loc": 85.5, "scale": 40.0}, -0.3, category="Purchase History", min_val=5.0),
        Column("total_spent", DTypes.NUMERIC, "numpy.normal", {"loc": 850, "scale": 620}, -0.45, category="Purchase History", min_val=5.0),
        Column("items_purchased_total", DTypes.NUMERIC, "numpy.poisson", {"lam": 16}, -0.35, category="Purchase History", min_val=1.0),
        Column("avg_items_per_order", DTypes.NUMERIC, "numpy.normal", {"loc": 2.8, "scale": 1.2}, -0.15, category="Purchase History", min_val=1.0),

        Column("return_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 1.0}, 0.8, category="Returns & Cart Behavior", min_val=0.0, max_val=1.0),
        Column("return_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 2}, 0.45, category="Returns & Cart Behavior", min_val=0.0),
        Column("cart_abandon_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 1.0}, 0.65, category="Returns & Cart Behavior", min_val=0.0, max_val=1.0),
        Column("checkout_abandon_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.8}, 0.5, category="Returns & Cart Behavior", min_val=0.0, max_val=1.0),
        Column("refund_amount_total", DTypes.NUMERIC, "numpy.normal", {"loc": 65, "scale": 85}, 0.35, category="Returns & Cart Behavior", min_val=0.0),

        Column("app_usage_hours", DTypes.NUMERIC, "numpy.normal", {"loc": 11.2, "scale": 5.8}, -0.55, category="Marketing & Engagement", min_val=0.0),
        Column("email_open_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 1.0}, -0.45, category="Marketing & Engagement", min_val=0.0, max_val=1.0),
        Column("push_notifications_enabled", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.45}, -0.3, category="Marketing & Engagement"),
        Column("used_coupon_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 1.0}, -0.2, category="Marketing & Engagement", min_val=0.0, max_val=1.0),
        Column("loyalty_points_balance", DTypes.NUMERIC, "numpy.poisson", {"lam": 650}, -0.65, category="Marketing & Engagement", min_val=0.0),
        Column("product_reviews_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, -0.4, category="Marketing & Engagement", min_val=0.0),
        Column("preferred_category", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["electronics", "fashion", "home", "grocery", "beauty"]}, 0.0, category="Marketing & Engagement"),

        Column("support_chats_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.5}, 0.4, category="Service & Delivery", min_val=0.0),
        Column("delivery_delay_complaints", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.6}, 0.75, category="Service & Delivery", min_val=0.0),
        Column("avg_delivery_days", DTypes.NUMERIC, "numpy.normal", {"loc": 3.8, "scale": 1.5}, 0.3, category="Service & Delivery", min_val=1.0),
        Column("satisfaction_rating", DTypes.NUMERIC, "numpy.uniform", {"low": 1.0, "high": 5.0}, -0.7, category="Service & Delivery", min_val=1.0, max_val=5.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["order_count", "total_spent"], "matrix": [[1.0, 0.75], [0.75, 1.0]]},
        {"type": "correlation", "columns": ["avg_order_value", "total_spent"], "matrix": [[1.0, 0.60], [0.60, 1.0]]},
        {"type": "conditional", "target": "days_since_last_order", "condition_col": "app_usage_hours", "condition_val": ">20", "dist": "uniform", "params": {"low": 1, "high": 30}},
        {"type": "interaction", "conditions": [{"col": "days_since_last_order", "op": ">", "val": "180"}, {"col": "loyalty_points_balance", "op": "<", "val": "150"}], "churn_boost": 1.1},
        {"type": "interaction", "conditions": [{"col": "return_rate", "op": ">", "val": "0.45"}, {"col": "delivery_delay_complaints", "op": ">=", "val": "2"}], "churn_boost": 0.85},
        {"type": "nonlinear", "column": "days_since_last_order", "transform": "step", "params": {"threshold": 90, "below": -0.5, "above": 0.85}}
    ]
}

# ==============================================================================
# 5. GAMING DOMAIN PRESET (NEW)
# ==============================================================================
GAMING_PRESET = {
    "name": "Gaming",
    "description": "Live-service, multiplayer and mobile gaming player retention dataset tracking session frequency, monetization, matchmaking friction, and social clans.",
    "target_churn_rate": 0.28,
    "noise_level": 0.3,
    "columns": [
        Column("player_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Player Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 24, "scale": 7}, -0.1, category="Player Profile", min_val=13.0, max_val=70.0),
        Column("platform", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["PC", "PlayStation", "Xbox", "Mobile", "Nintendo Switch"]}, 0.0, category="Player Profile"),
        Column("region", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["NA-East", "NA-West", "EU-Central", "Asia-East", "LATAM", "Oceania"]}, 0.0, category="Player Profile"),
        Column("account_age_days", DTypes.NUMERIC, "numpy.uniform", {"low": 7, "high": 1200}, -0.3, category="Player Profile", min_val=1.0),
        
        Column("daily_playtime_mins", DTypes.NUMERIC, "numpy.normal", {"loc": 85, "scale": 55}, -0.85, category="Engagement & Activity", min_val=0.0),
        Column("sessions_last_7d", DTypes.NUMERIC, "numpy.poisson", {"lam": 12}, -0.9, category="Engagement & Activity", min_val=0.0),
        Column("days_since_last_login", DTypes.NUMERIC, "numpy.uniform", {"low": 0, "high": 60}, 1.2, category="Engagement & Activity", min_val=0.0),
        Column("session_length_avg_mins", DTypes.NUMERIC, "numpy.normal", {"loc": 42, "scale": 20}, -0.4, category="Engagement & Activity", min_val=5.0),
        Column("weekend_play_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.2, "high": 0.9}, -0.15, category="Engagement & Activity", min_val=0.0, max_val=1.0),

        Column("level_achieved", DTypes.NUMERIC, "numpy.poisson", {"lam": 35}, -0.65, category="Skill & Progression", min_val=1.0),
        Column("win_rate_pct", DTypes.NUMERIC, "numpy.normal", {"loc": 49.5, "scale": 8.0}, -0.7, category="Skill & Progression", min_val=10.0, max_val=90.0),
        Column("ranked_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Unranked"]}, -0.4, category="Skill & Progression"),
        Column("kd_ratio", DTypes.NUMERIC, "numpy.normal", {"loc": 1.15, "scale": 0.45}, -0.35, category="Skill & Progression", min_val=0.1),
        Column("tutorial_completed", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.88}, -0.6, category="Skill & Progression"),

        Column("battle_pass_active", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.34}, -0.8, category="Monetization"),
        Column("total_iap_usd", DTypes.NUMERIC, "numpy.lognormal", {"mean": 2.8, "sigma": 1.6}, -0.75, category="Monetization", min_val=0.0),
        Column("cosmetic_skins_owned", DTypes.NUMERIC, "numpy.poisson", {"lam": 8}, -0.4, category="Monetization", min_val=0.0),
        Column("virtual_currency_balance", DTypes.NUMERIC, "numpy.poisson", {"lam": 450}, -0.2, category="Monetization", min_val=0.0),

        Column("clan_member", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.44}, -0.85, category="Social & Network"),
        Column("friend_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 9}, -0.6, category="Social & Network", min_val=0.0),
        Column("toxic_reports_received", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.4}, 0.5, category="Social & Network", min_val=0.0),
        Column("voice_chat_used", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.32}, -0.3, category="Social & Network"),
        Column("server_ping_ms", DTypes.NUMERIC, "numpy.normal", {"loc": 58, "scale": 32}, 0.45, category="Technical Quality", min_val=10.0),
        Column("game_crashes_last_month", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.8}, 0.75, category="Technical Quality", min_val=0.0),
        Column("rage_quits_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.9}, 0.65, category="Technical Quality", min_val=0.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["daily_playtime_mins", "total_iap_usd"], "matrix": [[1.0, 0.65], [0.65, 1.0]]},
        {"type": "correlation", "columns": ["level_achieved", "cosmetic_skins_owned"], "matrix": [[1.0, 0.70], [0.70, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "server_ping_ms", "op": ">=", "val": "120"}, {"col": "game_crashes_last_month", "op": ">=", "val": "3"}], "churn_boost": 1.25},
        {"type": "interaction", "conditions": [{"col": "clan_member", "op": "==", "val": True}, {"col": "friend_count", "op": ">=", "val": "8"}], "churn_boost": -0.9},
        {"type": "interaction", "conditions": [{"col": "days_since_last_login", "op": ">=", "val": "14"}, {"col": "battle_pass_active", "op": "==", "val": False}], "churn_boost": 1.4},
        {"type": "nonlinear", "column": "days_since_last_login", "transform": "step", "params": {"threshold": 21, "below": -0.6, "above": 1.3}}
    ]
}

# ==============================================================================
# 6. STREAMING & MEDIA (OTT) PRESET (NEW)
# ==============================================================================
STREAMING_PRESET = {
    "name": "Streaming",
    "description": "Video and audio on-demand (VOD/OTT) subscription churn modeling watch-time decay, catalog discovery, buffering latency, and household sharing.",
    "target_churn_rate": 0.23,
    "noise_level": 0.3,
    "columns": [
        Column("subscriber_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Subscriber Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 37, "scale": 13}, -0.1, category="Subscriber Profile", min_val=18.0, max_val=85.0),
        Column("household_profiles_count", DTypes.NUMERIC, "numpy.random.choice", {"a": [1, 2, 3, 4, 5], "p": [0.35, 0.30, 0.18, 0.12, 0.05]}, -0.6, category="Subscriber Profile", min_val=1.0, max_val=5.0),
        Column("primary_device", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Smart TV", "Streaming Stick", "Mobile Phone", "Tablet", "Web Browser"]}, 0.0, category="Subscriber Profile"),
        Column("kids_profile_active", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.36}, -0.35, category="Subscriber Profile"),
        
        Column("plan_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Ad-Supported", "Standard HD", "Premium 4K"]}, -0.5, category="Plan & Billing"),
        Column("billing_period", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Monthly", "Annual"]}, -0.7, category="Plan & Billing"),
        Column("tenure_months", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 60}, -0.75, category="Plan & Billing", min_val=1.0),
        Column("price_increase_in_last_60d", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.22}, 0.65, category="Plan & Billing"),
        Column("shared_outside_household", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.28}, 0.3, category="Plan & Billing"),
        Column("auto_renew_enabled", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.78}, -0.4, category="Plan & Billing"),

        Column("weekly_watch_hours", DTypes.NUMERIC, "numpy.normal", {"loc": 14.5, "scale": 9.2}, -0.9, category="Viewing Habits", min_val=0.0),
        Column("active_days_per_month", DTypes.NUMERIC, "numpy.poisson", {"lam": 16}, -0.85, category="Viewing Habits", min_val=0.0, max_val=31.0),
        Column("series_completion_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.1, "high": 1.0}, -0.7, category="Viewing Habits", min_val=0.0, max_val=1.0),
        Column("download_for_offline_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, -0.3, category="Viewing Habits", min_val=0.0),
        Column("rewatch_frequency_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.6}, -0.25, category="Viewing Habits", min_val=0.0, max_val=1.0),
        Column("binge_sessions_per_month", DTypes.NUMERIC, "numpy.poisson", {"lam": 4}, -0.45, category="Viewing Habits", min_val=0.0),

        Column("favorite_genre", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Drama", "Action & Thriller", "Comedy", "Documentary", "Sci-Fi", "Kids & Family", "Anime"]}, 0.0, category="Content & Discovery"),
        Column("search_abandon_rate", DTypes.NUMERIC, "numpy.uniform", {"low": 0.05, "high": 0.75}, 0.6, category="Content & Discovery", min_val=0.0, max_val=1.0),
        Column("watchlist_size", DTypes.NUMERIC, "numpy.poisson", {"lam": 14}, -0.5, category="Content & Discovery", min_val=0.0),
        Column("recently_finished_series", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.40}, 0.45, category="Content & Discovery"),
        Column("content_recommendation_clicks", DTypes.NUMERIC, "numpy.poisson", {"lam": 8}, -0.35, category="Content & Discovery", min_val=0.0),

        Column("buffering_events_per_hour", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.6}, 0.7, category="Stream Quality", min_val=0.0),
        Column("avg_bitrate_mbps", DTypes.NUMERIC, "numpy.normal", {"loc": 11.5, "scale": 4.2}, -0.3, category="Stream Quality", min_val=1.5),
        Column("app_crashes_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.5}, 0.55, category="Stream Quality", min_val=0.0),
        Column("support_tickets_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.4}, 0.5, category="Stream Quality", min_val=0.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["weekly_watch_hours", "active_days_per_month"], "matrix": [[1.0, 0.72], [0.72, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "recently_finished_series", "op": "==", "val": True}, {"col": "watchlist_size", "op": "<", "val": "3"}], "churn_boost": 1.1},
        {"type": "interaction", "conditions": [{"col": "buffering_events_per_hour", "op": ">=", "val": "2"}, {"col": "price_increase_in_last_60d", "op": "==", "val": True}], "churn_boost": 1.3},
        {"type": "interaction", "conditions": [{"col": "household_profiles_count", "op": ">=", "val": "3"}, {"col": "billing_period", "op": "==", "val": "Annual"}], "churn_boost": -0.85},
        {"type": "nonlinear", "column": "weekly_watch_hours", "transform": "step", "params": {"threshold": 4, "below": 1.2, "above": -0.45}}
    ]
}

# ==============================================================================
# 7. RIDE-HAILING & MOBILITY PRESET (NEW)
# ==============================================================================
RIDEHAILING_PRESET = {
    "name": "Ride-Hailing",
    "description": "Urban on-demand ride hailing rider dataset capturing surge pricing sensitivity, ETA waiting friction, driver cancellations, and multi-tier usage.",
    "target_churn_rate": 0.25,
    "noise_level": 0.3,
    "columns": [
        Column("rider_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Rider Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 33, "scale": 11}, 0.0, category="Rider Profile", min_val=18.0, max_val=80.0),
        Column("city_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Tier-1 Metro", "Tier-2 Urban", "Tier-3 Suburban"]}, 0.0, category="Rider Profile"),
        Column("membership_subscription", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.22}, -0.75, category="Rider Profile"),
        Column("months_active", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 48}, -0.3, category="Rider Profile", min_val=1.0),
        Column("default_payment_type", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Credit Card", "Apple Pay / Google Pay", "Debit Card", "Cash / Wallet"]}, 0.0, category="Rider Profile"),
        Column("corporate_business_account", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.18}, -0.5, category="Rider Profile"),

        Column("rides_last_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 7}, -0.85, category="Trip Activity", min_val=0.0),
        Column("lifetime_rides", DTypes.NUMERIC, "numpy.lognormal", {"mean": 3.6, "sigma": 1.2}, -0.4, category="Trip Activity", min_val=1.0),
        Column("avg_trip_distance_miles", DTypes.NUMERIC, "numpy.normal", {"loc": 6.8, "scale": 4.1}, -0.1, category="Trip Activity", min_val=0.5),
        Column("weekend_night_rides_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.8}, -0.15, category="Trip Activity", min_val=0.0, max_val=1.0),
        Column("preferred_ride_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Economy / Shared", "Standard", "Comfort", "XL / Black"]}, -0.2, category="Trip Activity"),
        Column("airport_trips_count_last_year", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, -0.35, category="Trip Activity", min_val=0.0),

        Column("avg_surge_multiplier", DTypes.NUMERIC, "numpy.normal", {"loc": 1.25, "scale": 0.25}, 0.7, category="Pricing & Surge", min_val=1.0),
        Column("surge_ride_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.05, "high": 0.65}, 0.6, category="Pricing & Surge", min_val=0.0, max_val=1.0),
        Column("promo_discount_usage_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.75}, 0.25, category="Pricing & Surge", min_val=0.0, max_val=1.0),
        Column("avg_fare_usd", DTypes.NUMERIC, "numpy.normal", {"loc": 22.5, "scale": 12.0}, -0.1, category="Pricing & Surge", min_val=4.0),
        Column("tipping_frequency_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.8}, -0.3, category="Pricing & Surge", min_val=0.0, max_val=1.0),

        Column("avg_wait_time_mins", DTypes.NUMERIC, "numpy.normal", {"loc": 5.4, "scale": 3.2}, 0.65, category="Service Experience", min_val=1.0),
        Column("rider_cancelled_trips_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.2}, 0.4, category="Service Experience", min_val=0.0),
        Column("driver_cancelled_trips_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.8}, 0.9, category="Service Experience", min_val=0.0),
        Column("avg_driver_rating_given", DTypes.NUMERIC, "numpy.uniform", {"low": 3.5, "high": 5.0}, -0.5, category="Service Experience", min_val=1.0, max_val=5.0),
        Column("rider_rating_received", DTypes.NUMERIC, "numpy.normal", {"loc": 4.85, "scale": 0.2}, -0.2, category="Service Experience", min_val=3.0, max_val=5.0),
        Column("customer_support_disputes", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.3}, 0.8, category="Service Experience", min_val=0.0),
        Column("app_open_to_request_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.3, "high": 0.95}, -0.4, category="Service Experience", min_val=0.0, max_val=1.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["rides_last_30d", "lifetime_rides"], "matrix": [[1.0, 0.65], [0.65, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "avg_wait_time_mins", "op": ">=", "val": "9"}, {"col": "avg_surge_multiplier", "op": ">=", "val": "1.5"}], "churn_boost": 1.2},
        {"type": "interaction", "conditions": [{"col": "driver_cancelled_trips_30d", "op": ">=", "val": "3"}], "churn_boost": 1.15},
        {"type": "interaction", "conditions": [{"col": "membership_subscription", "op": "==", "val": True}, {"col": "rides_last_30d", "op": ">=", "val": "10"}], "churn_boost": -0.9},
        {"type": "nonlinear", "column": "rides_last_30d", "transform": "step", "params": {"threshold": 2, "below": 1.0, "above": -0.5}}
    ]
}

# ==============================================================================
# 8. FITNESS & WELLNESS MEMBERSHIPS PRESET (NEW)
# ==============================================================================
FITNESS_PRESET = {
    "name": "Fitness",
    "description": "Gym, health club, and fitness studio recurring membership churn modeling visit frequency decay, group classes, PT sessions, and contract renewals.",
    "target_churn_rate": 0.27,
    "noise_level": 0.3,
    "columns": [
        Column("member_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Member Profile"),
        Column("age", DTypes.NUMERIC, "numpy.normal", {"loc": 32.5, "scale": 10.5}, -0.15, category="Member Profile", min_val=16.0, max_val=80.0),
        Column("gender", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Male", "Female", "Other"]}, 0.0, category="Member Profile"),
        Column("distance_to_gym_miles", DTypes.NUMERIC, "numpy.lognormal", {"mean": 1.2, "sigma": 0.6}, 0.55, category="Member Profile", min_val=0.2),
        Column("friend_member_referral", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.34}, -0.45, category="Member Profile"),

        Column("membership_tier", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Basic One-Club", "All-Access Multi-Club", "VIP / Premier", "Off-Peak"]}, -0.4, category="Membership & Contract"),
        Column("contract_duration_months", DTypes.NUMERIC, "numpy.random.choice", {"a": [1, 6, 12, 24], "p": [0.45, 0.15, 0.35, 0.05]}, -0.8, category="Membership & Contract", min_val=1.0),
        Column("monthly_fee_usd", DTypes.NUMERIC, "numpy.normal", {"loc": 59.0, "scale": 28.0}, 0.2, category="Membership & Contract", min_val=19.0),
        Column("tenure_months", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 48}, -0.6, category="Membership & Contract", min_val=1.0),
        Column("join_month_january", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.28}, 0.45, category="Membership & Contract"),
        Column("autopay_enabled", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.81}, -0.5, category="Membership & Contract"),

        Column("visits_last_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 6}, -1.1, category="Attendance & Routine", min_val=0.0),
        Column("visits_previous_month", DTypes.NUMERIC, "numpy.poisson", {"lam": 9}, -0.3, category="Attendance & Routine", min_val=0.0),
        Column("attendance_drop_pct", DTypes.NUMERIC, "numpy.uniform", {"low": -0.2, "high": 0.95}, 0.95, category="Attendance & Routine", min_val=-0.5, max_val=1.0),
        Column("avg_workout_mins", DTypes.NUMERIC, "numpy.normal", {"loc": 52, "scale": 18}, -0.25, category="Attendance & Routine", min_val=15.0),
        Column("weekend_visits_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.6}, -0.1, category="Attendance & Routine", min_val=0.0, max_val=1.0),
        Column("early_morning_attendee", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.26}, -0.35, category="Attendance & Routine"),

        Column("personal_trainer_sessions_used", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.2}, -0.7, category="Services & Classes", min_val=0.0),
        Column("group_classes_attended_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 2.5}, -0.75, category="Services & Classes", min_val=0.0),
        Column("locker_rental", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.18}, -0.35, category="Services & Classes"),
        Column("towel_service", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.25}, -0.2, category="Services & Classes"),
        Column("app_usage_checkins", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.55}, -0.3, category="Services & Classes"),
        Column("crowded_hours_complaint", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.12}, 0.5, category="Services & Classes"),
        Column("body_composition_scans", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.8}, -0.4, category="Services & Classes", min_val=0.0)
    ],
    "relationships": [
        {"type": "correlation", "columns": ["visits_last_30d", "visits_previous_month"], "matrix": [[1.0, 0.70], [0.70, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "attendance_drop_pct", "op": ">=", "val": "0.60"}, {"col": "contract_duration_months", "op": "==", "val": 1}], "churn_boost": 1.4},
        {"type": "interaction", "conditions": [{"col": "group_classes_attended_30d", "op": ">=", "val": "4"}, {"col": "personal_trainer_sessions_used", "op": ">=", "val": "2"}], "churn_boost": -1.0},
        {"type": "interaction", "conditions": [{"col": "distance_to_gym_miles", "op": ">=", "val": "5"}, {"col": "visits_last_30d", "op": "<=", "val": "2"}], "churn_boost": 1.2},
        {"type": "nonlinear", "column": "visits_last_30d", "transform": "step", "params": {"threshold": 3, "below": 1.2, "above": -0.6}}
    ]
}

# ==============================================================================
# 9. EDTECH & ONLINE LEARNING PRESET (NEW)
# ==============================================================================
EDTECH_PRESET = {
    "name": "EdTech",
    "description": "Digital learning platform subscriber retention tracking course progress streaks, quiz pass rates, video completion velocity, and career goals.",
    "target_churn_rate": 0.25,
    "noise_level": 0.3,
    "columns": [
        Column("learner_id", DTypes.CATEGORICAL, "faker.uuid4", {}, 0.0, category="Learner Profile"),
        Column("learning_goal", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Career Switch", "Job Promotion", "Academic Degree", "Personal Curiosity", "Certification"]}, -0.2, category="Learner Profile"),
        Column("employment_status", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Employed Full-time", "Student", "Job Seeker", "Freelancer"]}, 0.0, category="Learner Profile"),
        Column("country", DTypes.CATEGORICAL, "faker.country", {}, 0.0, category="Learner Profile"),
        Column("account_age_months", DTypes.NUMERIC, "numpy.uniform", {"low": 1, "high": 36}, -0.3, category="Learner Profile", min_val=1.0),
        Column("highest_education_level", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["High School", "Bachelor's", "Master's", "Doctorate", "Self-Taught"]}, -0.1, category="Learner Profile"),

        Column("subscription_type", DTypes.CATEGORICAL, "faker.random_element", {"elements": ["Monthly All-Access", "Annual Pass", "Enterprise / Employer Sponsored", "Pay-Per-Course"]}, -0.65, category="Subscription & Plan"),
        Column("monthly_price_usd", DTypes.NUMERIC, "numpy.normal", {"loc": 39.0, "scale": 15.0}, 0.2, category="Subscription & Plan", min_val=15.0),
        Column("financial_aid_granted", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.14}, -0.3, category="Subscription & Plan"),
        Column("auto_renew", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.68}, -0.5, category="Subscription & Plan"),

        Column("active_days_last_30d", DTypes.NUMERIC, "numpy.poisson", {"lam": 10}, -0.9, category="Study Momentum", min_val=0.0, max_val=31.0),
        Column("study_streak_days", DTypes.NUMERIC, "numpy.poisson", {"lam": 4}, -0.85, category="Study Momentum", min_val=0.0),
        Column("weekly_study_hours", DTypes.NUMERIC, "numpy.normal", {"loc": 5.5, "scale": 3.8}, -0.7, category="Study Momentum", min_val=0.0),
        Column("video_watch_completion_pct", DTypes.NUMERIC, "numpy.uniform", {"low": 0.1, "high": 1.0}, -0.6, category="Study Momentum", min_val=0.0, max_val=1.0),
        Column("playback_speed_avg", DTypes.NUMERIC, "numpy.random.choice", {"a": [1.0, 1.25, 1.5, 1.75, 2.0], "p": [0.35, 0.25, 0.25, 0.1, 0.05]}, 0.0, category="Study Momentum", min_val=1.0, max_val=2.0),

        Column("courses_enrolled", DTypes.NUMERIC, "numpy.poisson", {"lam": 3}, 0.1, category="Courses & Progress", min_val=1.0),
        Column("courses_completed", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.2}, -0.65, category="Courses & Progress", min_val=0.0),
        Column("quiz_pass_rate_pct", DTypes.NUMERIC, "numpy.normal", {"loc": 76.0, "scale": 14.0}, -0.7, category="Courses & Progress", min_val=10.0, max_val=100.0),
        Column("assignments_submitted_ontime", DTypes.NUMERIC, "numpy.poisson", {"lam": 4}, -0.5, category="Courses & Progress", min_val=0.0),
        Column("certificates_earned", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.8}, -0.6, category="Courses & Progress", min_val=0.0),

        Column("forum_discussions_posted", DTypes.NUMERIC, "numpy.poisson", {"lam": 1.5}, -0.4, category="Community & Support", min_val=0.0),
        Column("mentor_sessions_attended", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.6}, -0.55, category="Community & Support", min_val=0.0),
        Column("mobile_learning_ratio", DTypes.NUMERIC, "numpy.uniform", {"low": 0.0, "high": 0.8}, -0.1, category="Community & Support", min_val=0.0, max_val=1.0),
        Column("support_inquiries_count", DTypes.NUMERIC, "numpy.poisson", {"lam": 0.5}, 0.35, category="Community & Support", min_val=0.0),
        Column("career_coaching_used", DTypes.BOOLEAN, "numpy.binomial", {"n": 1, "p": 0.22}, -0.45, category="Community & Support")
    ],
    "relationships": [
        {"type": "correlation", "columns": ["active_days_last_30d", "weekly_study_hours"], "matrix": [[1.0, 0.68], [0.68, 1.0]]},
        {"type": "correlation", "columns": ["courses_completed", "certificates_earned"], "matrix": [[1.0, 0.82], [0.82, 1.0]]},
        {"type": "interaction", "conditions": [{"col": "study_streak_days", "op": "==", "val": 0}, {"col": "active_days_last_30d", "op": "<=", "val": "2"}], "churn_boost": 1.3},
        {"type": "interaction", "conditions": [{"col": "quiz_pass_rate_pct", "op": "<", "val": "55"}, {"col": "assignments_submitted_ontime", "op": "==", "val": 0}], "churn_boost": 1.1},
        {"type": "interaction", "conditions": [{"col": "subscription_type", "op": "==", "val": "Enterprise / Employer Sponsored"}], "churn_boost": -0.9},
        {"type": "nonlinear", "column": "active_days_last_30d", "transform": "step", "params": {"threshold": 4, "below": 1.1, "above": -0.5}}
    ]
}

# ==============================================================================
# GLOBAL DOMAIN REGISTRY
# ==============================================================================
PRESETS = {
    "telecom": TELECOM_PRESET,
    "saas": SAAS_PRESET,
    "banking": BANKING_PRESET,
    "ecommerce": ECOMMERCE_PRESET,
    "gaming": GAMING_PRESET,
    "streaming": STREAMING_PRESET,
    "ridehailing": RIDEHAILING_PRESET,
    "fitness": FITNESS_PRESET,
    "edtech": EDTECH_PRESET
}
