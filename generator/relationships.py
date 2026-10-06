from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

class RelationshipEngine:
    """
    Advanced Engine to apply statistical relationships, Gaussian copulas,
    composite multi-condition interaction rules, and non-linear hazard curves.
    """

    def apply_correlations(self, df: pd.DataFrame, rules: list, noise_level: float, telemetry: list) -> pd.DataFrame:
        """
        Inject target correlations using the Iman-Conover rank-permutation copula method.
        Solves overlapping correlation rules jointly via connected components and positive semi-definite projection.
        Guarantees 100% preservation of unique values and marginal distribution shapes without resampling.
        """
        df_new = df.copy()
        n_samples = len(df_new)
        if n_samples < 2:
            return df_new

        # 1. Collect valid active correlation rules
        corr_rules = []
        for idx, rule in enumerate(rules):
            if rule.get('type') != 'correlation':
                continue
            if rule.get('enabled') is False:
                telemetry.append({
                    "index": idx,
                    "type": "correlation",
                    "enabled": False,
                    "active": False,
                    "missing_columns": [],
                    "rows_matched": 0,
                    "target_r": 0.5,
                    "realized_r": 0.0,
                    "warning": "Rule muted by user."
                })
                continue
            cols = rule.get('columns', [])
            missing = [c for c in cols if c not in df_new.columns]
            if missing or len(cols) < 2:
                telemetry.append({
                    "index": idx,
                    "type": "correlation",
                    "enabled": True,
                    "active": False,
                    "missing_columns": missing,
                    "rows_matched": 0,
                    "target_r": 0.5,
                    "realized_r": 0.0,
                    "warning": f"Inactive: required column(s) {missing} not included in dataset."
                })
                continue

            target_r = 0.5
            if rule.get('matrix') is not None and len(rule['matrix']) >= 2 and len(rule['matrix'][0]) >= 2:
                target_r = float(rule['matrix'][0][1])
            elif rule.get('parameter') is not None:
                target_r = float(rule['parameter'])

            corr_rules.append({
                "index": idx,
                "cols": cols,
                "target_r": target_r,
                "rule": rule
            })

        if not corr_rules:
            return df_new

        # 2. Build graph of connected components for overlapping columns
        adj = {}
        for cr in corr_rules:
            c1, c2 = cr['cols'][0], cr['cols'][1]
            adj.setdefault(c1, set()).add(c2)
            adj.setdefault(c2, set()).add(c1)

        visited = set()
        components = []
        for node in adj:
            if node not in visited:
                comp = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        # 3. Process each connected component jointly
        for comp in components:
            k = len(comp)
            if k < 2:
                continue

            # Build k x k correlation matrix
            R = np.eye(k, dtype=float)
            for cr in corr_rules:
                c1, c2 = cr['cols'][0], cr['cols'][1]
                if c1 in comp and c2 in comp:
                    i = comp.index(c1)
                    j = comp.index(c2)
                    R[i, j] = cr['target_r']
                    R[j, i] = cr['target_r']

            # Estimate unspecified cross-correlations via product-path heuristic
            for i in range(k):
                for j in range(k):
                    if i != j and R[i, j] == 0.0:
                        for m in range(k):
                            if R[i, m] != 0.0 and R[m, j] != 0.0:
                                R[i, j] = R[i, m] * R[m, j]
                                R[j, i] = R[i, j]
                                break

            # Project to nearest positive semi-definite (PSD) correlation matrix
            R = (R + R.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, 1e-4)
            R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d = np.sqrt(np.diag(R_psd))
            d = np.where(d <= 0, 1.0, d)
            R_psd = R_psd / np.outer(d, d)
            R_psd = (R_psd + R_psd.T) / 2.0
            np.fill_diagonal(R_psd, 1.0)

            try:
                L = cholesky(R_psd, lower=True)
            except Exception:
                R_psd += 1e-3 * np.eye(k)
                L = cholesky(R_psd, lower=True)

            # Generate standard normal scores and correlate
            Z = np.random.randn(n_samples, k)
            Z_corr = Z @ L.T

            if noise_level > 0:
                Z_noise = np.random.randn(*Z_corr.shape)
                blend = min(0.12 * noise_level, 0.25)
                Z_corr = (1.0 - blend) * Z_corr + blend * Z_noise

            # Apply Iman-Conover rank permutation (100% preservation of values)
            for i, col_name in enumerate(comp):
                orig_vals = pd.to_numeric(df_new[col_name], errors='coerce').fillna(0.0).to_numpy()
                sorted_orig = np.sort(orig_vals)
                ranks = np.argsort(np.argsort(Z_corr[:, i]))
                df_new[col_name] = sorted_orig[ranks]

        # 4. Record realized telemetry for each correlation rule
        for cr in corr_rules:
            c1, c2 = cr['cols'][0], cr['cols'][1]
            v1 = pd.to_numeric(df_new[c1], errors='coerce').to_numpy()
            v2 = pd.to_numeric(df_new[c2], errors='coerce').to_numpy()
            if len(v1) > 1 and np.std(v1) > 0 and np.std(v2) > 0:
                realized_r = round(float(np.corrcoef(v1, v2)[0, 1]), 3)
                spearman_rho = round(float(stats.spearmanr(v1, v2)[0]), 3)
            else:
                realized_r = 0.0
                spearman_rho = 0.0

            telemetry.append({
                "index": cr["index"],
                "type": "correlation",
                "enabled": True,
                "active": True,
                "missing_columns": [],
                "rows_matched": n_samples,
                "target_r": cr["target_r"],
                "realized_r": realized_r,
                "realized_spearman": spearman_rho,
                "warning": ""
            })

        return df_new


    def _eval_condition(self, df: pd.DataFrame, col: str, val: Any, op: Optional[str] = None) -> np.ndarray:
        """
        Robustly evaluate conditions supporting >=, <=, !=, ==, >, < with automatic type conversion.
        Guarantees numeric 1 matches "1", boolean True matches "True"/"true", and categorical strings match cleanly.
        """
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        series = df[col]
        raw_target = val
        effective_op = op or "=="

        if isinstance(val, str):
            val_str = val.strip()
            # Extract operator prefix if embedded in the value string
            for candidate_op in [">=", "<=", "!=", "==", ">", "<"]:
                if val_str.startswith(candidate_op):
                    effective_op = candidate_op
                    raw_target = val_str[len(candidate_op):].strip()
                    break

        # Type coercion based on target column's dtype
        # NOTE: is_bool_dtype MUST be checked before is_numeric_dtype because pandas classifies bools as numeric subtypes.
        target = raw_target
        if pd.api.types.is_bool_dtype(series):
            if isinstance(raw_target, str):
                low = raw_target.strip().lower()
                if low in ('true', '1', 'yes', 't'):
                    target = True
                elif low in ('false', '0', 'no', 'f'):
                    target = False
            elif isinstance(raw_target, (int, float)):
                target = bool(raw_target)
        elif pd.api.types.is_numeric_dtype(series):
            try:
                target = float(raw_target)
            except (ValueError, TypeError):
                target = raw_target
        else:
            # String / categorical
            if isinstance(raw_target, str):
                target = raw_target.strip()

        try:
            if effective_op == "==":
                return (series == target).to_numpy()
            elif effective_op == "!=":
                return (series != target).to_numpy()
            elif effective_op == ">=":
                return (series >= target).to_numpy()
            elif effective_op == "<=":
                return (series <= target).to_numpy()
            elif effective_op == ">":
                return (series > target).to_numpy()
            elif effective_op == "<":
                return (series < target).to_numpy()
        except TypeError:
            # Fallback for incompatible types: compare string representations
            s_str = series.astype(str).str.strip()
            t_str = str(target).strip()
            if effective_op == "==":
                return (s_str == t_str).to_numpy()
            elif effective_op == "!=":
                return (s_str != t_str).to_numpy()
            elif effective_op == ">=":
                return (s_str >= t_str).to_numpy()
            elif effective_op == "<=":
                return (s_str <= t_str).to_numpy()
            elif effective_op == ">":
                return (s_str > t_str).to_numpy()
            elif effective_op == "<":
                return (s_str < t_str).to_numpy()

        return (series == target).to_numpy()

    def apply_conditional_distributions(self, df: pd.DataFrame, rules: list, noise_level: float, telemetry: list) -> pd.DataFrame:
        """
        Overwrite target column values based on condition.
        Updates telemetry per rule.
        """
        df_new = df.copy()
        for idx, rule in enumerate(rules):
            if rule.get('type') != 'conditional':
                continue
            if rule.get('enabled') is False:
                telemetry.append({
                    "index": idx,
                    "type": "conditional",
                    "enabled": False,
                    "active": False,
                    "missing_columns": [],
                    "rows_matched": 0,
                    "warning": "Rule muted by user."
                })
                continue

            target = rule.get('target')
            cond_col = rule.get('condition_col')
            cond_val = rule.get('condition_val')
            dist = rule.get('dist')
            params = rule.get('params', {})

            missing = [c for c in [target, cond_col] if not c or c not in df_new.columns]
            if missing:
                telemetry.append({
                    "index": idx,
                    "type": "conditional",
                    "enabled": True,
                    "active": False,
                    "missing_columns": missing,
                    "rows_matched": 0,
                    "warning": f"Inactive: required column(s) {missing} not included in dataset."
                })
                continue

            mask = self._eval_condition(df_new, cond_col, cond_val)
            n_affected = int(mask.sum())
            warning = "Rule active but matched 0 rows." if n_affected == 0 else ""

            if n_affected > 0:
                if dist == 'normal':
                    loc = float(params.get('loc', 0.0))
                    scale = float(params.get('scale', 1.0)) * (1.0 + 0.1 * noise_level)
                    df_new.loc[mask, target] = np.random.normal(loc, max(scale, 1e-4), n_affected)
                elif dist == 'uniform':
                    low = float(params.get('low', 0.0))
                    high = float(params.get('high', 1.0))
                    df_new.loc[mask, target] = np.random.uniform(low, high, n_affected)
                elif dist == 'poisson':
                    lam = max(float(params.get('lam', 1.0)), 0.1)
                    df_new.loc[mask, target] = np.random.poisson(lam, n_affected)
                elif dist == 'choice':
                    elements = params.get('elements', [True, False])
                    values = np.random.choice(elements, n_affected)
                    try:
                        values = values.astype(df_new[target].dtype)
                    except (ValueError, TypeError):
                        pass
                    df_new.loc[mask, target] = values

            telemetry.append({
                "index": idx,
                "type": "conditional",
                "enabled": True,
                "active": True,
                "missing_columns": [],
                "rows_matched": n_affected,
                "warning": warning
            })

        return df_new

    def apply_interaction_effects(self, df: pd.DataFrame, rules: list, telemetry: list) -> np.ndarray:
        """
        Return churn boosts for interaction rules.
        Strictly prevents multi-condition degradation: if ANY condition column is missing,
        the rule will NOT fire on remaining columns.
        """
        boosts = np.zeros(len(df))
        for idx, rule in enumerate(rules):
            if rule.get('type') != 'interaction':
                continue
            if rule.get('enabled') is False:
                telemetry.append({
                    "index": idx,
                    "type": "interaction",
                    "enabled": False,
                    "active": False,
                    "missing_columns": [],
                    "rows_matched": 0,
                    "warning": "Rule muted by user."
                })
                continue

            raw_conds = rule.get('conditions', {})
            boost = float(rule.get('churn_boost', 0.0))

            # Extract all required condition columns
            required_cols = []
            if isinstance(raw_conds, list):
                required_cols = [item.get('col') for item in raw_conds if item.get('col')]
            elif isinstance(raw_conds, dict):
                required_cols = list(raw_conds.keys())

            missing = [c for c in required_cols if c not in df.columns]
            if missing or not required_cols:
                telemetry.append({
                    "index": idx,
                    "type": "interaction",
                    "enabled": True,
                    "active": False,
                    "missing_columns": missing,
                    "rows_matched": 0,
                    "warning": f"Inactive: required condition column(s) {missing} not included in dataset."
                })
                continue

            mask = np.ones(len(df), dtype=bool)
            if isinstance(raw_conds, list):
                for item in raw_conds:
                    c = item.get('col')
                    op = item.get('op', '==')
                    v = item.get('val')
                    mask = mask & self._eval_condition(df, c, v, op=op)
            elif isinstance(raw_conds, dict):
                for col, val in raw_conds.items():
                    mask = mask & self._eval_condition(df, col, val)

            n_affected = int(mask.sum())
            if n_affected > 0:
                boosts[mask] += boost

            warning = "Rule active but matched 0 rows." if n_affected == 0 else ""
            telemetry.append({
                "index": idx,
                "type": "interaction",
                "enabled": True,
                "active": True,
                "missing_columns": [],
                "rows_matched": n_affected,
                "warning": warning
            })

        return boosts

    def apply_nonlinear_effects(self, df: pd.DataFrame, rules: list, telemetry: list) -> np.ndarray:
        """
        Return churn boosts for non-linear rules (step threshold, u-curve, exponential decay).
        """
        boosts = np.zeros(len(df))
        for idx, rule in enumerate(rules):
            if rule.get('type') != 'nonlinear':
                continue
            if rule.get('enabled') is False:
                telemetry.append({
                    "index": idx,
                    "type": "nonlinear",
                    "enabled": False,
                    "active": False,
                    "missing_columns": [],
                    "rows_matched": 0,
                    "warning": "Rule muted by user."
                })
                continue

            col = rule.get('column')
            if not col or col not in df.columns:
                telemetry.append({
                    "index": idx,
                    "type": "nonlinear",
                    "enabled": True,
                    "active": False,
                    "missing_columns": [col] if col else [],
                    "rows_matched": 0,
                    "warning": f"Inactive: column '{col}' not included in dataset."
                })
                continue

            transform = rule.get('transform')
            params = rule.get('params', {})

            try:
                col_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                if transform == 'step':
                    thresh = float(params.get('threshold', 0.0))
                    below = float(params.get('below', 0.0))
                    above = float(params.get('above', 0.0))
                    boosts += np.where(col_vals < thresh, below, above)
                elif transform == 'u_curve':
                    center = float(params.get('center', 0.0))
                    scale = float(params.get('scale', 0.001))
                    boosts += scale * ((col_vals - center) ** 2)
                elif transform == 'decay':
                    rate = float(params.get('rate', 0.1))
                    scale = float(params.get('scale', 1.0))
                    boosts += scale * np.exp(-rate * np.maximum(col_vals, 0.0))

                telemetry.append({
                    "index": idx,
                    "type": "nonlinear",
                    "enabled": True,
                    "active": True,
                    "missing_columns": [],
                    "rows_matched": len(df),
                    "warning": ""
                })
            except Exception as e:
                telemetry.append({
                    "index": idx,
                    "type": "nonlinear",
                    "enabled": True,
                    "active": False,
                    "missing_columns": [],
                    "rows_matched": 0,
                    "warning": f"Evaluation error: {str(e)}"
                })

        return boosts

    def apply_all(self, df: pd.DataFrame, relationship_rules: list, noise_level: float) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], list]:
        """
        Orchestrate all relationships safely and collect execution telemetry.
        Returns: (df, churn_contributions, rule_telemetry)
        """
        telemetry = []
        
        # 1. Correlations (Iman-Conover rank permutation + connected components)
        df = self.apply_correlations(df, relationship_rules, noise_level, telemetry)
        
        # 2. Conditionals
        df = self.apply_conditional_distributions(df, relationship_rules, noise_level, telemetry)

        # 3. Interactions & Non-linear transforms
        interactions = self.apply_interaction_effects(df, relationship_rules, telemetry)
        nonlinears = self.apply_nonlinear_effects(df, relationship_rules, telemetry)

        churn_contributions = {
            'interactions': interactions,
            'nonlinears': nonlinears
        }

        # Sort telemetry by original rule index
        telemetry.sort(key=lambda x: x["index"])

        return df, churn_contributions, telemetry


