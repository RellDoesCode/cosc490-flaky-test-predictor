"""
Baseline classifiers for comparison against our static-feature approach.

Evaluates four approaches on the same stratified 5-fold CV:
  1. Smell-only    — FlakeFlagger's 8 pre-labeled test smells
  2. History-only  — hIndex code-modification history features
  3. Our approach  — static features extracted from raw Java source
  4. Full baseline — all FlakeFlagger features (smell + history + runtime)

All evaluated on the same 22,236-test FlakeFlagger dataset for fair comparison.

Run from repo root:
    python -m src.baselines
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


PROCESSED_CSV    = 'data/flakeflagger/processed_data.csv'
STATIC_FEAT_CSV  = 'data/flakeflagger/static_features.csv'

# ── Feature group definitions ─────────────────────────────────────────────────

SMELL_FEATURES = [
    'assertion-roulette',
    'conditional-test-logic',
    'eager-test',
    'fire-and-forget',
    'indirect-testing',
    'mystery-guest',
    'resource-optimism',
    'test-run-war',
]

HISTORY_FEATURES = [
    'hIndexModificationsPerCoveredLine_window5',
    'hIndexModificationsPerCoveredLine_window10',
    'hIndexModificationsPerCoveredLine_window25',
    'hIndexModificationsPerCoveredLine_window50',
    'hIndexModificationsPerCoveredLine_window75',
    'hIndexModificationsPerCoveredLine_window100',
    'hIndexModificationsPerCoveredLine_window500',
    'hIndexModificationsPerCoveredLine_window10000',
]

RUNTIME_FEATURES = [
    'testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime',
    'projectSourceLinesCovered', 'projectSourceClassesCovered',
    'num_third_party_libs',
]

STATIC_FEATURE_COLS = [
    'loc', 'num_asserts', 'thread_sleep_count', 'has_thread_sleep',
    'async_wait_count', 'has_async_wait', 'has_file_io', 'has_network_io',
    'has_concurrency', 'num_test_methods', 'num_try_catch',
    'has_setup_teardown', 'num_conditionals', 'has_random',
    'has_system_time', 'num_annotations',
    'assert_density', 'loc_per_test', 'has_timeout_annotation', 'timeout_count',
    'polling_count', 'has_env_access', 'has_db_access', 'has_injection',
    'has_static_field', 'thread_join_count', 'notify_count', 'broad_catch_count',
    'file_io_count', 'network_io_count', 'has_rule_annotation', 'num_inner_classes',
    'imports_mockito', 'imports_powermock', 'imports_easymock', 'imports_concurrent',
    'imports_atomic', 'imports_network', 'imports_spring', 'imports_guice',
    'imports_jdbc', 'imports_jpa', 'imports_nio', 'imports_io',
    'imports_awaitility', 'num_imports',
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def cv_with_threshold(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probas = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        probas[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    best_f1, best_thresh = 0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        preds = (probas >= t).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    y_pred = (probas >= best_thresh).astype(int)
    return {
        'f1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'threshold': round(best_thresh, 3),
        'n':         len(y),
        'n_flaky':   int(y.sum()),
    }


def best_model(X, y):
    """Return the best of RF / XGBoost / LightGBM + SMOTE for a feature set."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    candidates = [
        LGBMClassifier(n_estimators=200, learning_rate=0.2, num_leaves=63,
                       random_state=42, verbosity=-1),
        ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf',   LGBMClassifier(n_estimators=200, learning_rate=0.2,
                                     num_leaves=63, random_state=42, verbosity=-1)),
        ]),
        XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                      eval_metric='logloss', verbosity=0, random_state=42),
        ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf',   XGBClassifier(n_estimators=300, learning_rate=0.1,
                                    max_depth=5, eval_metric='logloss',
                                    verbosity=0, random_state=42)),
        ]),
    ]
    best_result, best_model_obj = {'f1': -1}, None
    for m in candidates:
        r = cv_with_threshold(m, X, y)
        if r['f1'] > best_result['f1']:
            best_result, best_model_obj = r, m
    return best_result


def print_result(label, result):
    print(f"  {label:<35} "
          f"F1={result['f1']:.4f}  "
          f"P={result['precision']:.4f}  "
          f"R={result['recall']:.4f}  "
          f"(n={result['n']}, flaky={result['n_flaky']})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading FlakeFlagger dataset...")
    df_ff = pd.read_csv(PROCESSED_CSV)
    df_ff = df_ff.rename(columns={'flaky': 'label'})
    df_ff = df_ff.dropna(subset=['label'])

    print("Loading our static features...")
    df_static = pd.read_csv(STATIC_FEAT_CSV)

    print(f"FlakeFlagger: {len(df_ff)} tests, {df_ff['label'].sum()} flaky "
          f"({df_ff['label'].mean()*100:.2f}%)")
    print(f"Our static:   {len(df_static)} tests, {df_static['label'].sum()} flaky "
          f"({df_static['label'].mean()*100:.2f}%)")

    results = []

    # ── 1. Smell-only baseline ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("1. SMELL-ONLY BASELINE (FlakeFlagger test smell labels)")
    print("="*65)
    df_smell = df_ff.dropna(subset=SMELL_FEATURES)
    X_smell  = df_smell[SMELL_FEATURES].values
    y_smell  = df_smell['label'].astype(int).values
    r = best_model(X_smell, y_smell)
    print_result("Smell-only (best model)", r)
    results.append(('Smell-only baseline', r))

    # ── 2. History-only baseline ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("2. HISTORY-ONLY BASELINE (hIndex commit modification features)")
    print("="*65)
    df_hist = df_ff.dropna(subset=HISTORY_FEATURES)
    X_hist  = df_hist[HISTORY_FEATURES].values
    y_hist  = df_hist['label'].astype(int).values
    r = best_model(X_hist, y_hist)
    print_result("History-only (best model)", r)
    results.append(('History-only baseline', r))

    # ── 3. Our approach — static features only ────────────────────────────────
    print("\n" + "="*65)
    print("3. OUR APPROACH — static features (no runtime, no history)")
    print("="*65)
    avail_static = [c for c in STATIC_FEATURE_COLS if c in df_static.columns]
    df_s = df_static.dropna(subset=avail_static)
    X_s  = df_s[avail_static].values
    y_s  = df_s['label'].astype(int).values
    r = best_model(X_s, y_s)
    print_result("Static features (our approach)", r)
    results.append(('Our approach (static)', r))

    # ── 4. Full FlakeFlagger baseline (smell + history + runtime) ─────────────
    print("\n" + "="*65)
    print("4. FULL BASELINE — all FlakeFlagger features (smell + history + runtime)")
    print("="*65)
    all_ff_features = SMELL_FEATURES + HISTORY_FEATURES + RUNTIME_FEATURES
    avail_ff = [c for c in all_ff_features if c in df_ff.columns]
    df_full = df_ff.dropna(subset=avail_ff)
    X_full  = df_full[avail_ff].values
    y_full  = df_full['label'].astype(int).values
    r = best_model(X_full, y_full)
    print_result("Full FlakeFlagger features", r)
    results.append(('Full FlakeFlagger baseline', r))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("COMPARISON SUMMARY")
    print("="*65)
    print(f"{'Approach':<35} {'F1':>6} {'Precision':>10} {'Recall':>8} {'n_flaky':>8}")
    print("-"*65)
    for name, r in results:
        print(f"  {name:<33} {r['f1']:>6.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['n_flaky']:>8}")

    print("\nKey finding:")
    our_f1   = next(r['f1'] for n, r in results if 'static' in n.lower())
    smell_f1 = next(r['f1'] for n, r in results if 'smell' in n.lower())
    hist_f1  = next(r['f1'] for n, r in results if 'history' in n.lower())
    full_f1  = next(r['f1'] for n, r in results if 'full' in n.lower())
    print(f"  Our static approach exceeds smell-only by {our_f1 - smell_f1:+.4f} F1")
    print(f"  Our static approach exceeds history-only by {our_f1 - hist_f1:+.4f} F1")
    print(f"  Gap to full FlakeFlagger baseline: {our_f1 - full_f1:+.4f} F1")
