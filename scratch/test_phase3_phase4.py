import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import pandas as pd
from scipy import stats

from generator.schema import Column, DTypes
from generator.presets import PRESETS
from generator.engine import generate_dataset
from generator.relationships import RelationshipEngine
from generator.domain_invariants import apply_domain_invariants, verify_domain_invariants
from app import app


class TestPhase3Phase4(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # =========================================================================
    # PHASE 4: RANK-PRESERVING CORRELATION ENGINE (IMAN-CONOVER & JOINT GRAPH)
    # =========================================================================

    def test_iman_conover_preserves_100_percent_unique_continuous_values(self):
        """
        Verify that Iman-Conover rank permutation strictly preserves 100% of unique continuous values,
        unlike sampling with replacement which causes marginal distribution collapse.
        """
        n = 3000
        # 1. Direct engine test verifying exact multiset equality
        raw_x = np.random.RandomState(42).randn(n)
        raw_y = np.random.RandomState(43).randn(n)
        df_raw = pd.DataFrame({"x": raw_x, "y": raw_y})

        engine = RelationshipEngine()
        corr_rule = {
            "type": "correlation",
            "columns": ["x", "y"],
            "parameter": 0.80,
            "enabled": True
        }
        df_corr, _, _ = engine.apply_all(df_raw, [corr_rule], noise_level=0.0)

        # The multiset of values in each column MUST be 100% identical before and after
        np.testing.assert_array_equal(np.sort(df_raw["x"].to_numpy()), np.sort(df_corr["x"].to_numpy()),
            err_msg="Iman-Conover must strictly preserve the exact multiset of continuous values for x!")
        np.testing.assert_array_equal(np.sort(df_raw["y"].to_numpy()), np.sort(df_corr["y"].to_numpy()),
            err_msg="Iman-Conover must strictly preserve the exact multiset of continuous values for y!")

        # 2. End-to-end generate_dataset test with high-cardinality continuous features
        c1 = Column("feat_a", DTypes.NUMERIC, "numpy.uniform", {"low": 10.0, "high": 100000.0})
        c2 = Column("feat_b", DTypes.NUMERIC, "numpy.uniform", {"low": 20.0, "high": 200000.0})
        rule = {
            "type": "correlation",
            "columns": ["feat_a", "feat_b"],
            "parameter": 0.75,
            "enabled": True
        }
        config = {
            "num_rows": n,
            "seed": 42,
            "columns": [c1, c2],
            "relationships": [rule],
            "noise_level": 0.0
        }
        df = generate_dataset(config)

        # All 3,000 values must remain unique (zero sampling-with-replacement collapse)
        self.assertEqual(df["feat_a"].nunique(), n, f"feat_a should have {n} unique values, got {df['feat_a'].nunique()}")
        self.assertEqual(df["feat_b"].nunique(), n, f"feat_b should have {n} unique values, got {df['feat_b'].nunique()}")

        # Target r=0.75: verify realized rank correlation (Spearman rho)
        spearman_rho = stats.spearmanr(df["feat_a"], df["feat_b"])[0]
        self.assertGreater(spearman_rho, 0.68, f"Expected high rank correlation, got {spearman_rho}")
        self.assertLess(spearman_rho, 0.85, f"Expected rank correlation near 0.75, got {spearman_rho}")

    def test_joint_connected_component_preserves_overlapping_correlations(self):
        """
        Verify that overlapping correlation rules (A-B and B-C) are solved jointly via
        connected-component PSD projection rather than sequentially destroying one another.
        """
        n = 3000
        config = {
            "preset_name": "ecommerce",
            "num_rows": n,
            "seed": 123,
            "noise_level": 0.05
        }
        df = generate_dataset(config)

        # E-commerce defines:
        # 1. order_count <-> total_spent (target 0.75)
        # 2. avg_order_value <-> total_spent (target 0.60)
        corr_order_spent = stats.spearmanr(df["order_count"], df["total_spent"])[0]
        corr_aov_spent = stats.spearmanr(df["avg_order_value"], df["total_spent"])[0]

        self.assertGreater(corr_order_spent, 0.50,
            f"Overlapping correlation order_count <-> total_spent collapsed! Realized rho = {corr_order_spent}")
        self.assertGreater(corr_aov_spent, 0.45,
            f"Overlapping correlation avg_order_value <-> total_spent collapsed! Realized rho = {corr_aov_spent}")

        # Check telemetry recorded in df.attrs
        rule_stats = df.attrs.get("rule_stats", [])
        corr_stats = [s for s in rule_stats if s.get("type") == "correlation"]
        self.assertEqual(len(corr_stats), 2)
        for cs in corr_stats:
            self.assertTrue(cs["active"])
            self.assertIn("target_r", cs)
            self.assertIn("realized_r", cs)
            self.assertIn("realized_spearman", cs)
            self.assertGreater(cs["realized_r"], 0.40)

    # =========================================================================
    # PHASE 3: DOMAIN INVARIANTS ACROSS ALL 9 PRESETS
    # =========================================================================

    def test_telecom_domain_invariants(self):
        """Verify Telecom service gating and physical charge bounds."""
        df = generate_dataset({"preset_name": "telecom", "num_rows": 2000, "seed": 42})
        report = verify_domain_invariants(df, "telecom")

        self.assertTrue(report["is_compliant"], f"Telecom invariants violated: {report['violation_details']}")
        self.assertEqual(report["total_violations"], 0)

        # Invariant checks:
        no_net = df["internet_service"].str.lower().isin(["no", "none", "no internet service"])
        if "streaming_tv" in df.columns:
            self.assertTrue((df.loc[no_net, "streaming_tv"] == "No internet service").all())
        if "online_security" in df.columns:
            self.assertTrue((df.loc[no_net, "online_security"] == "No internet service").all())

        no_phone = df["phone_service"].astype(str).str.lower().isin(["false", "0", "no", "no phone service"])
        if "multiple_lines" in df.columns:
            self.assertTrue((df.loc[no_phone, "multiple_lines"] == "No phone service").all())

    def test_edtech_domain_invariants(self):
        """Verify EdTech courses completed <= enrolled and certificates <= completed."""
        df = generate_dataset({"preset_name": "edtech", "num_rows": 2000, "seed": 42})
        report = verify_domain_invariants(df, "edtech")

        self.assertTrue(report["is_compliant"], f"EdTech invariants violated: {report['violation_details']}")
        self.assertEqual(report["total_violations"], 0)
        self.assertTrue((df["courses_completed"] <= df["courses_enrolled"]).all())
        if "certificates_earned" in df.columns:
            self.assertTrue((df["certificates_earned"] <= df["courses_completed"]).all())

    def test_ecommerce_domain_invariants(self):
        """Verify E-Commerce chronology and bounds."""
        df = generate_dataset({"preset_name": "ecommerce", "num_rows": 2000, "seed": 42})
        report = verify_domain_invariants(df, "ecommerce")

        self.assertTrue(report["is_compliant"], f"E-Commerce invariants violated: {report['violation_details']}")
        self.assertEqual(report["total_violations"], 0)
        self.assertTrue((df["return_count"] <= df["order_count"]).all())

    def test_banking_domain_invariants(self):
        """Verify Banking age vs account age and card consistency."""
        df = generate_dataset({"preset_name": "banking", "num_rows": 2000, "seed": 42})
        if "age" in df.columns and "account_age_years" in df.columns:
            self.assertTrue((df["age"] >= df["account_age_years"] + 17.5).all())
        if "credit_card_tier" in df.columns:
            no_card = df["credit_card_tier"].astype(str).str.lower() == "none"
            if "credit_card_balance" in df.columns:
                self.assertTrue((df.loc[no_card, "credit_card_balance"] == 0.0).all())

    def test_saas_domain_invariants(self):
        """Verify SaaS free tier zeroing and contract value relations."""
        df = generate_dataset({"preset_name": "saas", "num_rows": 2000, "seed": 42})
        if "subscription_tier" in df.columns:
            free_tier = df["subscription_tier"].astype(str).str.lower().isin(["free", "trial"])
            if "monthly_contract_value" in df.columns:
                self.assertTrue((df.loc[free_tier, "monthly_contract_value"] == 0.0).all())
            if "annual_contract_value" in df.columns:
                self.assertTrue((df.loc[free_tier, "annual_contract_value"] == 0.0).all())

    def test_ridehailing_domain_invariants(self):
        """Verify Ride-Hailing lifetime rides >= 30d rides."""
        df = generate_dataset({"preset_name": "ridehailing", "num_rows": 2000, "seed": 42})
        report = verify_domain_invariants(df, "ridehailing")
        self.assertTrue(report["is_compliant"], f"Ridehailing invariants violated: {report['violation_details']}")
        self.assertEqual(report["total_violations"], 0)
        self.assertTrue((df["lifetime_rides"] >= df["rides_last_30d"]).all())

    def test_all_nine_presets_100_percent_compliant(self):
        """Verify that all 9 built-in presets achieve 100% domain invariant compliance."""
        for preset_name in PRESETS:
            df = generate_dataset({"preset_name": preset_name, "num_rows": 1000, "seed": 99})
            report = verify_domain_invariants(df, preset_name)
            self.assertTrue(report["is_compliant"],
                f"Preset '{preset_name}' has invariant violations: {report['violation_details']}")
            self.assertEqual(report["total_violations"], 0,
                f"Preset '{preset_name}' expected 0 violations, got {report['total_violations']}")
            self.assertEqual(report["compliant_records_pct"], 100.0)

    # =========================================================================
    # PREVIEW API TELEMETRY VERIFICATION
    # =========================================================================

    def test_preview_api_returns_invariants_and_correlation_telemetry(self):
        """Verify /api/preview returns invariants_report and rule_stats with correlation metrics."""
        payload = {
            "preset_name": "ecommerce",
            "preview_rows": 50,
            "target_churn_rate": 0.25,
            "noise_level": 0.1
        }
        res = self.client.post("/api/preview", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Invariants report
        self.assertIn("invariants_report", data["stats"])
        inv = data["stats"]["invariants_report"]
        self.assertTrue(inv["is_compliant"])
        self.assertEqual(inv["compliant_records_pct"], 100.0)

        # Rule stats
        rule_stats = data["stats"]["rule_stats"]
        corr_rules = [r for r in rule_stats if r.get("type") == "correlation"]
        self.assertGreater(len(corr_rules), 0)
        for cr in corr_rules:
            self.assertIn("target_r", cr)
            self.assertIn("realized_r", cr)
            self.assertIn("realized_spearman", cr)


if __name__ == "__main__":
    unittest.main()
