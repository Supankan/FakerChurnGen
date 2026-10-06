import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import pandas as pd
from generator.schema import Column, DTypes, VALID_DTYPES
from generator.engine import generate_dataset, resolve_generator, ALLOWED_GENERATORS
from generator.relationships import RelationshipEngine
from app import app, MAX_GENERATE_ROWS

class TestPhase1Phase2(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # =========================================================================
    # PHASE 1: SECURITY & BOUNDARY ENFORCEMENT
    # =========================================================================

    def test_p0_params_size_override_neutralized(self):
        """Verify that params.size cannot hijack the dataset row count."""
        col = Column("test_num", DTypes.NUMERIC, "numpy.uniform", {"low": 10, "high": 20, "size": 9999})
        config = {
            "num_rows": 25,
            "columns": [col],
            "relationships": []
        }
        df = generate_dataset(config)
        self.assertEqual(len(df), 25, "Row count must be strictly num_rows (25), not overridden by params.size (9999)")
        self.assertEqual(len(df["test_num"]), 25)

    def test_generator_allowlist_blocks_unauthorized_reflection(self):
        """Verify that unauthorized generator strings are rejected."""
        with self.assertRaises(ValueError):
            resolve_generator("os.system")
        with self.assertRaises(ValueError):
            resolve_generator("numpy.random.system")
        with self.assertRaises(ValueError):
            resolve_generator("faker.__class__")

        # Legitimate generators must succeed
        self.assertTrue(callable(resolve_generator("numpy.uniform")))
        self.assertTrue(callable(resolve_generator("faker.city")))
        self.assertTrue(callable(resolve_generator("faker.uuid4")))

    def test_column_schema_validation(self):
        """Verify strict column schema validation and sanitization."""
        # Reserved name 'churn'
        with self.assertRaises(ValueError):
            Column.from_dict({"name": "churn", "dtype": "numeric"})
        with self.assertRaises(ValueError):
            Column.from_dict({"name": "CHURN", "dtype": "numeric"})

        # Empty name
        with self.assertRaises(ValueError):
            Column.from_dict({"name": "   ", "dtype": "numeric"})

        # Invalid dtype
        with self.assertRaises(ValueError):
            Column.from_dict({"name": "my_col", "dtype": "unsupported_dtype"})

        # Params stripping 'size'
        col = Column.from_dict({"name": "good_col", "dtype": "numeric", "params": {"size": 5000, "low": 1}})
        self.assertNotIn("size", col.params)
        self.assertEqual(col.params.get("low"), 1)

    def test_api_boundary_caps_and_error_handling(self):
        """Verify row ceilings and structured error handling across API endpoints."""
        # /api/preview clamping (max 200)
        res = self.client.post("/api/preview", json={"preset_name": "telecom", "preview_rows": 10000})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["stats"]["num_rows"], 200, "Preview must be clamped to 200 rows")
        self.assertIn("rule_stats", data["stats"])

        # /api/train_baseline model validation
        res_bad_model = self.client.post("/api/train_baseline", json={"preset_name": "telecom", "model_type": "deep_net"})
        self.assertEqual(res_bad_model.status_code, 400)
        self.assertIn("Invalid 'model_type'", res_bad_model.get_json()["error"])

        # /api/train_baseline malformed train_rows
        res_bad_rows = self.client.post("/api/train_baseline", json={"preset_name": "telecom", "train_rows": "invalid_number"})
        self.assertEqual(res_bad_rows.status_code, 400)
        self.assertIn("Invalid 'train_rows'", res_bad_rows.get_json()["error"])

        # /api/generate max ceiling (500k)
        res_overflow = self.client.post("/api/generate", json={"preset_name": "telecom", "num_rows": 600000})
        self.assertEqual(res_overflow.status_code, 400)
        self.assertIn("exceeds the maximum in-memory export ceiling", res_overflow.get_json()["error"])

    # =========================================================================
    # PHASE 2: TYPED RULE ENGINE & INTEGRITY AUDITING
    # =========================================================================

    def test_typed_rule_matching_numeric(self):
        """Verify numeric equality matching when value sent as string '1'."""
        engine = RelationshipEngine()
        df = pd.DataFrame({"tier": [1, 2, 1, 3, 1]})
        
        # When op is '==' and val is string "1" (as sent by UI)
        mask = engine._eval_condition(df, "tier", "1", op="==")
        self.assertEqual(list(mask), [True, False, True, False, True], "Numeric equality must match string '1'")

        # Operator prefix inside string: ">= 2"
        mask_op = engine._eval_condition(df, "tier", ">= 2")
        self.assertEqual(list(mask_op), [False, True, False, True, False])

    def test_typed_rule_matching_boolean(self):
        """Verify boolean equality matching when value sent as string 'True' or 'false'."""
        engine = RelationshipEngine()
        df = pd.DataFrame({"is_vip": [True, False, True, False]})
        
        mask_true = engine._eval_condition(df, "is_vip", "True", op="==")
        self.assertEqual(list(mask_true), [True, False, True, False], "Boolean equality must match string 'True'")

        mask_false = engine._eval_condition(df, "is_vip", "false", op="==")
        self.assertEqual(list(mask_false), [False, True, False, True], "Boolean equality must match string 'false'")

    def test_multi_condition_and_does_not_silently_degrade(self):
        """Verify that an interaction rule with a missing column does NOT fire on remaining columns."""
        engine = RelationshipEngine()
        df = pd.DataFrame({
            "feature_a": [1, 1, 0, 1],
            "feature_b": [10, 20, 30, 40]
        })
        telemetry = []
        rules = [
            {
                "type": "interaction",
                "conditions": [
                    {"col": "feature_a", "op": "==", "val": "1"},
                    {"col": "feature_MISSING", "op": "==", "val": "999"}
                ],
                "churn_boost": 2.5,
                "enabled": True
            }
        ]

        boosts = engine.apply_interaction_effects(df, rules, telemetry)
        
        # Boosts must be 0 for all rows (must NOT fire just because feature_a was 1)
        self.assertEqual(list(boosts), [0.0, 0.0, 0.0, 0.0], "Rule with missing column must NOT fire on partial match")
        self.assertEqual(len(telemetry), 1)
        self.assertFalse(telemetry[0]["active"], "Rule must be marked inactive")
        self.assertIn("feature_MISSING", telemetry[0]["missing_columns"])
        self.assertEqual(telemetry[0]["rows_matched"], 0)

    def test_rounding_before_scoring(self):
        """Verify that floating point features are rounded to 2 decimals before scoring."""
        col = Column("amount", DTypes.NUMERIC, "numpy.uniform", {"low": 10.12345, "high": 50.98765})
        config = {
            "num_rows": 50,
            "columns": [col],
            "relationships": []
        }
        df = generate_dataset(config)
        
        for val in df["amount"]:
            # Float representation check: str(val) decimal places <= 2
            val_str = f"{val:.6f}".rstrip("0")
            parts = val_str.split(".")
            if len(parts) > 1:
                self.assertLessEqual(len(parts[1]), 2, f"Value {val} has more than 2 decimal places")

    def test_rule_telemetry_attached_to_dataframe(self):
        """Verify that generate_dataset preserves rule_stats in df.attrs."""
        config = {
            "preset_name": "telecom",
            "num_rows": 50
        }
        df = generate_dataset(config)
        self.assertIn("rule_stats", df.attrs)
        stats = df.attrs["rule_stats"]
        self.assertGreater(len(stats), 0, "Rule telemetry must contain stats for preset rules")
        for r in stats:
            self.assertIn("type", r)
            self.assertIn("active", r)
            self.assertIn("rows_matched", r)

if __name__ == "__main__":
    unittest.main()
