from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

_ID_COL_PATTERNS = {'id', 'customer_id', 'user_id', 'player_id', 'subscriber_id', 'rider_id', 'member_id', 'learner_id'}

def train_and_evaluate_baseline(df: pd.DataFrame, model_type: str = "hist_gb") -> Dict[str, Any]:
    """
    Trains a baseline classifier on the generated dataset and returns
    comprehensive ML validation metrics, confusion matrix, and feature importances.
    """
    if 'churn' not in df.columns or len(df) < 30:
        raise ValueError("Dataset must contain 'churn' column and have at least 30 samples to evaluate.")

    # 1. Feature selection: drop ID columns and target
    feature_cols = [c for c in df.columns if c != 'churn' and c.lower() not in _ID_COL_PATTERNS]
    if not feature_cols:
        raise ValueError("No valid predictive features found to train model.")

    X = df[feature_cols].copy()
    y = df['churn'].values.astype(int)

    # If only 1 class is present, cannot train classifier
    if len(np.unique(y)) < 2:
        raise ValueError("Target 'churn' has only one class in the sample. Adjust churn rate or row count.")

    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            X[col] = X[col].astype(str)

    # 2. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Model training & pipeline
    if model_type == "logistic_regression":
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        model = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Feature importances via permutation on test set
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=3, random_state=42, scoring='roc_auc'
        )
        raw_importances = np.maximum(perm.importances_mean, 0.0)

    else:
        # Default: HistGradientBoostingClassifier (blazing fast, native support)
        cat_indices = [feature_cols.index(c) for c in categorical_cols]
        
        # Encode categoricals with OrdinalEncoder
        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        clf = HistGradientBoostingClassifier(
            categorical_features=cat_indices if cat_indices else None,
            max_iter=100,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        # Fast permutation importance on test subset
        eval_sample_size = min(len(X_test), 500)
        X_test_sub = X_test.iloc[:eval_sample_size]
        y_test_sub = y_test[:eval_sample_size]
        
        perm = permutation_importance(
            clf, X_test_sub, y_test_sub, n_repeats=3, random_state=42, scoring='roc_auc'
        )
        raw_importances = np.maximum(perm.importances_mean, 0.0)

    # 4. Compute ML metrics
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    total_test = len(y_test)
    cm_dict = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn_pct": round(float(tn / total_test * 100), 1),
        "fp_pct": round(float(fp / total_test * 100), 1),
        "fn_pct": round(float(fn / total_test * 100), 1),
        "tp_pct": round(float(tp / total_test * 100), 1),
    }

    # 5. Top 10 feature importances normalized to percentage
    sum_imp = float(np.sum(raw_importances))
    if sum_imp > 0:
        norm_importances = [(raw_importances[i] / sum_imp) * 100 for i in range(len(feature_cols))]
    else:
        norm_importances = [0.0] * len(feature_cols)

    feature_ranking = []
    for i, col in enumerate(feature_cols):
        feature_ranking.append({
            "feature": col,
            "importance": round(float(raw_importances[i]), 4),
            "pct": round(float(norm_importances[i]), 1)
        })

    # Sort descending
    feature_ranking.sort(key=lambda x: x["importance"], reverse=True)
    top_10_features = feature_ranking[:10]

    # 6. Diagnostic narrative
    top_names = [f["feature"] for f in top_10_features[:3]]
    top_pct_sum = sum(f["pct"] for f in top_10_features[:3])
    
    if roc_auc >= 0.80:
        quality = "High Discriminative Separation"
        narrative = f"Strong signal recovery. The top drivers ({', '.join(top_names)}) account for {top_pct_sum:.0f}% of relative importance, closely reflecting the causal logic defined."
    elif roc_auc >= 0.65:
        quality = "Moderate Realistic Noise"
        narrative = f"Balanced signal-to-noise ratio. Primary predictive drivers are {', '.join(top_names)}."
    else:
        quality = "High Stochastic Dispersion"
        narrative = "Weak classification signal. Consider reducing statistical noise level or amplifying feature churn weights."

    return {
        "model_type": model_type,
        "sample_size": len(df),
        "test_size": total_test,
        "metrics": {
            "roc_auc": round(roc_auc, 3),
            "pr_auc": round(pr_auc, 3),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "confusion_matrix": cm_dict,
        "feature_importances": top_10_features,
        "diagnostic": {
            "quality": quality,
            "narrative": narrative,
            "top_driver": top_10_features[0]["feature"] if top_10_features else "N/A"
        }
    }
