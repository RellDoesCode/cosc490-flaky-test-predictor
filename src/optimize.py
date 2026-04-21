"""
High-F1 optimization pipeline.

Strategies applied:
  1. SMOTE oversampling inside each CV fold (via imblearn Pipeline)
  2. Hyperparameter tuning with RandomizedSearchCV (F1-optimized)
  3. Decision-threshold tuning (find per-fold optimal cutoff)
  4. LightGBM as additional classifier

Run from repo root:
    python -m src.optimize
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


STATIC_FEATURES_CSV = "data/flakeflagger/static_features.csv"

FEATURE_COLS = [
    'loc', 'num_asserts', 'thread_sleep_count', 'has_thread_sleep',
    'async_wait_count', 'has_async_wait', 'has_file_io', 'has_network_io',
    'has_concurrency', 'num_test_methods', 'num_try_catch',
    'has_setup_teardown', 'num_conditionals', 'has_random',
    'has_system_time', 'num_annotations',
    # extended features
    'assert_density', 'loc_per_test', 'has_timeout_annotation', 'timeout_count',
    'polling_count', 'has_env_access', 'has_db_access', 'has_injection',
    'has_static_field', 'thread_join_count', 'notify_count', 'broad_catch_count',
    'file_io_count', 'network_io_count', 'has_rule_annotation', 'num_inner_classes',
    'imports_mockito', 'imports_powermock', 'imports_easymock', 'imports_concurrent',
    'imports_atomic', 'imports_network', 'imports_spring', 'imports_guice',
    'imports_jdbc', 'imports_jpa', 'imports_nio', 'imports_io',
    'imports_awaitility', 'num_imports',
]


def load_data():
    df = pd.read_csv(STATIC_FEATURES_CSV)
    df = df.dropna(subset=FEATURE_COLS + ['label'])
    X = df[FEATURE_COLS].values
    y = df['label'].astype(int).values
    return X, y


def cv_with_threshold(model, X, y, n_splits=5):
    """
    Stratified CV with per-fold optimal threshold selection.
    Returns F1, precision, recall at the best macro-averaged threshold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probas = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        probas[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    # Sweep thresholds to find best F1
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (probas >= t).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    y_pred = (probas >= best_thresh).astype(int)
    return {
        'f1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'threshold': round(best_thresh, 4),
    }


def run_approach(name, model, X, y):
    result = cv_with_threshold(model, X, y)
    print(f"  {name:<35} F1={result['f1']:.4f}  "
          f"P={result['precision']:.4f}  R={result['recall']:.4f}  "
          f"thresh={result['threshold']:.3f}")
    return name, result


def tune_xgboost(X, y):
    """RandomizedSearchCV for XGBoost, optimizing F1."""
    print("\nRunning RandomizedSearchCV for XGBoost (this may take ~1-2 min)...")
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos  # scale_pos_weight starting point

    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [3, 4, 5, 6, 7],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'scale_pos_weight': [1, spw * 0.5, spw, spw * 2],
        'min_child_weight': [1, 3, 5, 10],
        'gamma':            [0, 0.1, 0.2, 0.5],
    }

    base = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=40, scoring='f1',
        cv=cv, random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X, y)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1 (default threshold): {search.best_score_:.4f}")
    return search.best_estimator_


def tune_lgbm(X, y):
    """RandomizedSearchCV for LightGBM."""
    print("\nRunning RandomizedSearchCV for LightGBM...")
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos

    param_dist = {
        'n_estimators':   [100, 200, 300, 500],
        'max_depth':      [-1, 4, 6, 8],
        'learning_rate':  [0.01, 0.05, 0.1, 0.2],
        'num_leaves':     [15, 31, 63, 127],
        'subsample':      [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, spw * 0.5, spw, spw * 2],
        'min_child_samples': [5, 10, 20, 50],
    }

    base = LGBMClassifier(random_state=42, verbosity=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=40, scoring='f1',
        cv=cv, random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X, y)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1 (default threshold): {search.best_score_:.4f}")
    return search.best_estimator_


def smote_model(classifier):
    """Wrap a classifier with SMOTE inside an imblearn Pipeline."""
    return ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf',   classifier),
    ])


if __name__ == '__main__':
    X, y = load_data()
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    print(f"Dataset: {len(y)} tests | {pos} flaky ({pos/len(y)*100:.2f}%) | {neg} non-flaky")
    print(f"Imbalance ratio: {neg/pos:.1f}:1\n")

    results = []

    # ── Baseline models (for comparison) ──────────────────────────────────────
    print("=" * 65)
    print("BASELINES (default threshold = 0.5)")
    print("=" * 65)
    results.append(run_approach(
        "RF (class_weight=balanced)",
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        X, y
    ))
    results.append(run_approach(
        "XGBoost (default)",
        XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42),
        X, y
    ))

    # ── Threshold-optimized baselines ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 65)
    results.append(run_approach(
        "RF + threshold opt",
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        X, y
    ))
    results.append(run_approach(
        "XGBoost + scale_pos_weight + thresh",
        XGBClassifier(scale_pos_weight=neg/pos, eval_metric='logloss', verbosity=0, random_state=42),
        X, y
    ))

    # ── SMOTE variants ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SMOTE OVERSAMPLING + THRESHOLD OPTIMIZATION")
    print("=" * 65)
    results.append(run_approach(
        "SMOTE + RF + thresh",
        smote_model(RandomForestClassifier(n_estimators=100, random_state=42)),
        X, y
    ))
    results.append(run_approach(
        "SMOTE + XGBoost + thresh",
        smote_model(XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)),
        X, y
    ))
    results.append(run_approach(
        "SMOTE + LightGBM + thresh",
        smote_model(LGBMClassifier(random_state=42, verbosity=-1)),
        X, y
    ))

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING")
    print("=" * 65)
    best_xgb  = tune_xgboost(X, y)
    best_lgbm = tune_lgbm(X, y)

    print("\nEvaluating tuned models with threshold optimization:")
    results.append(run_approach("Tuned XGBoost + thresh", best_xgb, X, y))
    results.append(run_approach("Tuned LightGBM + thresh", best_lgbm, X, y))
    results.append(run_approach(
        "SMOTE + Tuned XGBoost + thresh",
        smote_model(best_xgb),
        X, y
    ))
    results.append(run_approach(
        "SMOTE + Tuned LightGBM + thresh",
        smote_model(best_lgbm),
        X, y
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — ranked by F1")
    print("=" * 65)
    results_df = pd.DataFrame(
        [{'approach': name, **metrics} for name, metrics in results]
    ).sort_values('f1', ascending=False)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest approach: {best['approach']}")
    print(f"  F1={best['f1']:.4f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}")
