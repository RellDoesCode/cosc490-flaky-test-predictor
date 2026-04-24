"""
Leave-One-Project-Out (LOPO) cross-project evaluation.

Trains on all projects except one, tests on the held-out project.
Repeats for every project that has at least one flaky test.
Reports per-project F1 / precision / recall and overall averages.

Run from repo root:
    python -m experiments.cross_project_tests
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


STATIC_FEATURES_CSV = "data/flakeflagger/static_features.csv"

FEATURE_COLS = [
    'loc', 'num_asserts', 'thread_sleep_count', 'has_thread_sleep',
    'async_wait_count', 'has_async_wait', 'has_file_io', 'has_network_io',
    'has_concurrency', 'num_test_methods', 'num_try_catch',
    'has_setup_teardown', 'num_conditionals', 'has_random',
    'has_system_time', 'num_annotations',
    'assert_density', 'loc_per_test', 'has_timeout_annotation', 'timeout_count',
    'polling_count', 'has_env_access', 'has_db_access', 'has_injection',
    'has_static_field', 'thread_join_count', 'notify_count', 'broad_catch_count',
    'file_io_count', 'network_io_count', 'has_rule_annotation', 'num_inner_classes',
]


def load_data():
    df = pd.read_csv(STATIC_FEATURES_CSV)
    df = df.dropna(subset=FEATURE_COLS + ['label', 'project'])
    df['label'] = df['label'].astype(int)
    return df


def evaluate_lopo(df, model_fn, model_name):
    projects = df['project'].unique()

    results = []
    skipped = []

    for held_out in sorted(projects):
        train_df = df[df['project'] != held_out]
        test_df  = df[df['project'] == held_out]

        # Skip if test project has no flaky tests (can't compute meaningful F1)
        if test_df['label'].sum() == 0:
            skipped.append(held_out)
            continue

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['label'].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df['label'].values

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1  = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        n_flaky = int(y_test.sum())
        n_total = len(y_test)

        results.append({
            'project':   held_out,
            'n_total':   n_total,
            'n_flaky':   n_flaky,
            'f1':        round(f1, 3),
            'precision': round(prec, 3),
            'recall':    round(rec, 3),
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Leave-One-Project-Out Results")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    if skipped:
        print(f"\nSkipped (no flaky tests in test set): {', '.join(skipped)}")

    avg_f1   = results_df['f1'].mean()
    avg_prec = results_df['precision'].mean()
    avg_rec  = results_df['recall'].mean()

    print(f"\nMacro-average across {len(results_df)} projects:")
    print(f"  Avg F1:        {avg_f1:.3f}")
    print(f"  Avg Precision: {avg_prec:.3f}")
    print(f"  Avg Recall:    {avg_rec:.3f}")

    return results_df


def rf_factory():
    return RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )


def xgb_factory():
    return XGBClassifier(eval_metric='logloss', verbosity=0)


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} tests, {df['label'].sum()} flaky, {df['project'].nunique()} projects")

    rf_results  = evaluate_lopo(df, rf_factory,  "Random Forest")
    xgb_results = evaluate_lopo(df, xgb_factory, "XGBoost")

    print("\n\nSummary comparison (macro-average F1):")
    print(f"  Random Forest: {rf_results['f1'].mean():.3f}")
    print(f"  XGBoost:       {xgb_results['f1'].mean():.3f}")
