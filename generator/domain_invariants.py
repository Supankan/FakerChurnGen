import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

def apply_domain_invariants(df: pd.DataFrame, preset_name: Optional[str] = None) -> pd.DataFrame:
    """
    Enforces real-world physical and business domain invariants across all 9 presets.
    Eliminates impossible customer records, fixes parent-child dependencies,
    and applies hierarchical service gates.
    """
    df_clean = df.copy()
    preset = preset_name.lower().strip() if preset_name else ""

    # =========================================================================
    # 1. TELECOM DOMAIN INVARIANTS
    # =========================================================================
    if preset == "telecom" or ("internet_service" in df_clean.columns and "total_charges" in df_clean.columns):
        # Hierarchical Service Gating: No Internet -> No Internet Addons
        if "internet_service" in df_clean.columns:
            no_internet_mask = df_clean["internet_service"].astype(str).str.strip().str.lower().isin(["no", "none", "no internet service"])
            addon_cols = [
                "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies"
            ]
            for col in addon_cols:
                if col in df_clean.columns:
                    df_clean.loc[no_internet_mask, col] = "No internet service"

            if "avg_monthly_gb_download" in df_clean.columns:
                df_clean.loc[no_internet_mask, "avg_monthly_gb_download"] = 0.0
            if "extra_data_charges" in df_clean.columns:
                df_clean.loc[no_internet_mask, "extra_data_charges"] = 0.0

        # Hierarchical Phone Gating: No Phone -> No Multiple Lines
        if "phone_service" in df_clean.columns and "multiple_lines" in df_clean.columns:
            no_phone_mask = df_clean["phone_service"].astype(str).str.strip().str.lower().isin(["false", "0", "no", "no phone service"])
            df_clean.loc[no_phone_mask, "multiple_lines"] = "No phone service"

        # Arithmetic Consistency: Total Charges vs Tenure * Monthly Charges
        if "total_charges" in df_clean.columns and "tenure_months" in df_clean.columns and "monthly_charges" in df_clean.columns:
            tenure = np.maximum(df_clean["tenure_months"].to_numpy(dtype=float), 1.0)
            monthly = np.maximum(df_clean["monthly_charges"].to_numpy(dtype=float), 1.0)
            base_expected = tenure * monthly
            current_total = pd.to_numeric(df_clean["total_charges"], errors="coerce").to_numpy(dtype=float)
            current_total = np.where(np.isnan(current_total), base_expected, current_total)
            # Bounded physical limits: total charges must be between 75% (discounts) and 135% (taxes/fees) of tenure * monthly
            min_allowed = base_expected * 0.75
            max_allowed = base_expected * 1.35 + 25.0
            df_clean["total_charges"] = np.clip(current_total, min_allowed, max_allowed)

    # =========================================================================
    # 2. EDTECH DOMAIN INVARIANTS
    # =========================================================================
    if preset == "edtech" or ("courses_enrolled" in df_clean.columns and "courses_completed" in df_clean.columns):
        if "courses_enrolled" in df_clean.columns and "courses_completed" in df_clean.columns:
            enrolled = np.maximum(df_clean["courses_enrolled"].to_numpy(dtype=float), 1.0)
            completed = np.maximum(df_clean["courses_completed"].to_numpy(dtype=float), 0.0)
            # Invariant: Cannot complete more courses than enrolled
            df_clean["courses_completed"] = np.minimum(completed, enrolled)

        if "certificates_earned" in df_clean.columns and "courses_completed" in df_clean.columns:
            certs = np.maximum(df_clean["certificates_earned"].to_numpy(dtype=float), 0.0)
            completed = df_clean["courses_completed"].to_numpy(dtype=float)
            # Invariant: Cannot earn more certificates than courses completed
            df_clean["certificates_earned"] = np.minimum(certs, completed)

        if "study_streak_days" in df_clean.columns and "active_days_last_30d" in df_clean.columns:
            streaks = np.maximum(df_clean["study_streak_days"].to_numpy(dtype=float), 0.0)
            active_days = np.maximum(df_clean["active_days_last_30d"].to_numpy(dtype=float), 0.0)
            # Invariant: Streak within the month cannot exceed active days
            df_clean["study_streak_days"] = np.minimum(streaks, active_days)

    # =========================================================================
    # 3. E-COMMERCE DOMAIN INVARIANTS
    # =========================================================================
    if preset == "ecommerce" or ("days_since_first_order" in df_clean.columns and "days_since_last_order" in df_clean.columns):
        if "days_since_first_order" in df_clean.columns and "days_since_last_order" in df_clean.columns:
            first_order = np.maximum(df_clean["days_since_first_order"].to_numpy(dtype=float), 1.0)
            last_order = np.maximum(df_clean["days_since_last_order"].to_numpy(dtype=float), 0.0)
            # Invariant: Last order date cannot precede first order date
            df_clean["days_since_last_order"] = np.minimum(last_order, first_order)

        if "return_count" in df_clean.columns and "order_count" in df_clean.columns:
            orders = np.maximum(df_clean["order_count"].to_numpy(dtype=float), 1.0)
            returns = np.maximum(df_clean["return_count"].to_numpy(dtype=float), 0.0)
            # Invariant: Cannot return more orders than placed
            df_clean["return_count"] = np.minimum(returns, orders)

        if "total_spent" in df_clean.columns and "order_count" in df_clean.columns and "avg_order_value" in df_clean.columns:
            orders = np.maximum(df_clean["order_count"].to_numpy(dtype=float), 1.0)
            aov = np.maximum(df_clean["avg_order_value"].to_numpy(dtype=float), 1.0)
            expected_spend = orders * aov
            current_spent = pd.to_numeric(df_clean["total_spent"], errors="coerce").to_numpy(dtype=float)
            current_spent = np.where(np.isnan(current_spent), expected_spend, current_spent)
            # Bound spend within 70% to 130% of order_count * avg_order_value
            df_clean["total_spent"] = np.clip(current_spent, expected_spend * 0.70, expected_spend * 1.30)

    # =========================================================================
    # 4. SAAS DOMAIN INVARIANTS
    # =========================================================================
    if preset == "saas" or ("monthly_recurring_revenue" in df_clean.columns and "plan_tier" in df_clean.columns):
        if "plan_tier" in df_clean.columns:
            free_tier_mask = df_clean["plan_tier"].astype(str).str.strip().str.lower() == "free"
            if "monthly_recurring_revenue" in df_clean.columns:
                df_clean.loc[free_tier_mask, "monthly_recurring_revenue"] = 0.0
            if "total_contract_value" in df_clean.columns:
                df_clean.loc[free_tier_mask, "total_contract_value"] = 0.0

        if "active_seats_pct" in df_clean.columns:
            df_clean["active_seats_pct"] = np.clip(df_clean["active_seats_pct"].to_numpy(dtype=float), 0.0, 1.0)

        if "total_contract_value" in df_clean.columns and "monthly_recurring_revenue" in df_clean.columns and "contract_months" in df_clean.columns:
            mrr = np.maximum(df_clean["monthly_recurring_revenue"].to_numpy(dtype=float), 0.0)
            months = np.maximum(df_clean["contract_months"].to_numpy(dtype=float), 1.0)
            min_tcv = mrr * months * 0.75
            df_clean["total_contract_value"] = np.maximum(df_clean["total_contract_value"].to_numpy(dtype=float), min_tcv)

    # =========================================================================
    # 5. BANKING DOMAIN INVARIANTS
    # =========================================================================
    if preset == "banking" or ("account_age_months" in df_clean.columns and "age" in df_clean.columns):
        if "age" in df_clean.columns and "account_age_months" in df_clean.columns:
            age = np.maximum(df_clean["age"].to_numpy(dtype=float), 18.0)
            acc_age = np.maximum(df_clean["account_age_months"].to_numpy(dtype=float), 0.0)
            # Invariant: Account cannot be opened prior to age 16 (age - 16)*12
            max_acc_age = np.maximum((age - 16.0) * 12.0, 1.0)
            df_clean["account_age_months"] = np.minimum(acc_age, max_acc_age)

        if "has_credit_card" in df_clean.columns and "card_tier" in df_clean.columns:
            no_card = df_clean["has_credit_card"].astype(str).str.strip().str.lower().isin(["false", "0", "no"])
            df_clean.loc[no_card, "card_tier"] = "None"

    # =========================================================================
    # 6. GAMING DOMAIN INVARIANTS
    # =========================================================================
    if preset == "gaming" or ("battle_pass_active" in df_clean.columns):
        if "daily_playtime_mins" in df_clean.columns and "session_length_avg_mins" in df_clean.columns:
            # Daily playtime cannot be less than single average session if played today
            daily = df_clean["daily_playtime_mins"].to_numpy(dtype=float)
            session = df_clean["session_length_avg_mins"].to_numpy(dtype=float)
            df_clean["daily_playtime_mins"] = np.maximum(daily, np.where(daily > 0, session, 0.0))

    # =========================================================================
    # 7. STREAMING DOMAIN INVARIANTS
    # =========================================================================
    if preset == "streaming" or ("plan_tier" in df_clean.columns and "billing_period" in df_clean.columns):
        if "active_days_per_month" in df_clean.columns:
            df_clean["active_days_per_month"] = np.clip(df_clean["active_days_per_month"].to_numpy(dtype=float), 0.0, 31.0)

    # =========================================================================
    # 8. RIDE-HAILING DOMAIN INVARIANTS
    # =========================================================================
    if preset == "ridehailing" or ("rides_last_30d" in df_clean.columns and "lifetime_rides" in df_clean.columns):
        if "rides_last_30d" in df_clean.columns and "lifetime_rides" in df_clean.columns:
            rides_30d = np.maximum(df_clean["rides_last_30d"].to_numpy(dtype=float), 0.0)
            lifetime = np.maximum(df_clean["lifetime_rides"].to_numpy(dtype=float), 0.0)
            # Invariant: Lifetime rides must be >= rides in last 30 days
            df_clean["lifetime_rides"] = np.maximum(lifetime, rides_30d)

    # =========================================================================
    # 9. FITNESS DOMAIN INVARIANTS
    # =========================================================================
    if preset == "fitness" or ("attendance_drop_pct" in df_clean.columns and "visits_last_30d" in df_clean.columns):
        if "visits_last_30d" in df_clean.columns and "visits_previous_month" in df_clean.columns and "attendance_drop_pct" in df_clean.columns:
            v_curr = np.maximum(df_clean["visits_last_30d"].to_numpy(dtype=float), 0.0)
            v_prev = np.maximum(df_clean["visits_previous_month"].to_numpy(dtype=float), 0.0)
            # Physically synchronize attendance_drop_pct: (v_prev - v_curr) / max(v_prev, 1)
            safe_prev = np.maximum(v_prev, 1.0)
            drop_ratio = np.where(v_prev > 0, (v_prev - v_curr) / safe_prev, 0.0)
            # Add small realistic noise (-0.05 to +0.05) and clamp [-0.5, 1.0]
            noise = np.random.normal(0.0, 0.02, len(df_clean))
            df_clean["attendance_drop_pct"] = np.clip(drop_ratio + noise, -0.5, 1.0)

    return df_clean


def verify_domain_invariants(df: pd.DataFrame, preset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies that domain invariants hold strictly across the dataset.
    Returns violation counts and percentage compliance.
    """
    preset = preset_name.lower().strip() if preset_name else ""
    violations = {}

    if preset == "telecom" or ("total_charges" in df.columns and "tenure_months" in df.columns and "monthly_charges" in df.columns):
        tenure = df["tenure_months"].to_numpy(dtype=float)
        monthly = df["monthly_charges"].to_numpy(dtype=float)
        total = df["total_charges"].to_numpy(dtype=float)
        # Check total_charges < 0.5 * tenure * monthly
        severe_undercharges = np.sum(total < (tenure * monthly * 0.5))
        violations["telecom_severe_undercharges"] = int(severe_undercharges)

        if "internet_service" in df.columns and "streaming_tv" in df.columns:
            no_net = df["internet_service"].astype(str).str.strip().str.lower().isin(["no", "none", "no internet service"])
            illegal_streaming = np.sum(no_net & (df["streaming_tv"].astype(str).str.strip().str.lower() == "yes"))
            violations["telecom_illegal_streaming_without_net"] = int(illegal_streaming)

    if preset == "edtech" or ("courses_enrolled" in df.columns and "courses_completed" in df.columns):
        enrolled = df["courses_enrolled"].to_numpy(dtype=float)
        completed = df["courses_completed"].to_numpy(dtype=float)
        more_completed = np.sum(completed > enrolled)
        violations["edtech_completed_exceeds_enrolled"] = int(more_completed)

        if "certificates_earned" in df.columns:
            certs = df["certificates_earned"].to_numpy(dtype=float)
            more_certs = np.sum(certs > completed)
            violations["edtech_certs_exceed_completed"] = int(more_certs)

    if preset == "ecommerce" or ("days_since_first_order" in df.columns and "days_since_last_order" in df.columns):
        first = df["days_since_first_order"].to_numpy(dtype=float)
        last = df["days_since_last_order"].to_numpy(dtype=float)
        last_before_first = np.sum(last > first)
        violations["ecommerce_last_before_first"] = int(last_before_first)

    if preset == "ridehailing" or ("rides_last_30d" in df.columns and "lifetime_rides" in df.columns):
        r30 = df["rides_last_30d"].to_numpy(dtype=float)
        rlife = df["lifetime_rides"].to_numpy(dtype=float)
        violations["ridehailing_30d_exceeds_lifetime"] = int(np.sum(r30 > rlife))

    total_violations = sum(violations.values())
    total_checks = len(violations)
    is_compliant = (total_violations == 0)

    return {
        "is_compliant": is_compliant,
        "total_violations": total_violations,
        "violation_details": violations,
        "compliant_records_pct": 100.0 if is_compliant else round(float(1.0 - total_violations / (len(df) * max(total_checks, 1))) * 100, 2)
    }
