import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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
    'imports_mockito', 'imports_powermock', 'imports_easymock', 'imports_concurrent',
    'imports_atomic', 'imports_network', 'imports_spring', 'imports_guice',
    'imports_jdbc', 'imports_jpa', 'imports_nio', 'imports_io',
    'imports_awaitility', 'num_imports',
]

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    if 'flaky' in df.columns:
        df = df.rename(columns={'flaky': 'label'})

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"Note: skipping {len(missing)} missing features")

    df = df.dropna(subset=available_features + ['label', 'project'])
    df['label'] = df['label'].astype(int)
    df['project'] = df['project'].str.lower().str.strip()

    return df, available_features

def compute_project_stats(df):
    stats = df.groupby('project').agg({
        'label': 'mean',
        'has_concurrency': 'mean',
        'has_network_io': 'mean',
        'has_db_access': 'mean',
        'thread_sleep_count': 'mean',
        'async_wait_count': 'mean',
    }).rename(columns={'label': 'flaky_rate'})

    stats['io_score'] = stats['has_network_io'] + stats['has_db_access']
    return stats


def make_permutations(df, n_random=30):
    stats = compute_project_stats(df)
    perms = {}

    perms["Flaky High -> Low"] = stats.sort_values('flaky_rate', ascending=False).index.tolist()
    perms["Flaky Low -> High"] = stats.sort_values('flaky_rate').index.tolist()

    perms["Concurrency High -> Low"] = stats.sort_values('has_concurrency', ascending=False).index.tolist()
    perms["Concurrency Low -> High"] = stats.sort_values('has_concurrency').index.tolist()

    perms["IO Heavy -> Light"] = stats.sort_values('io_score', ascending=False).index.tolist()
    perms["IO Light -> Heavy"] = stats.sort_values('io_score').index.tolist()

    perms["Sleep Heavy -> Light"] = stats.sort_values('thread_sleep_count', ascending=False).index.tolist()

    for i in range(n_random):
        perms[f"Random {i+1}"] = stats.sample(frac=1, random_state=i).index.tolist()

    return perms

def build_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )


# -----------------------------
# Threshold tuning
# -----------------------------
def find_best_threshold(model, X_val, y_val):
    if len(np.unique(y_val)) < 2:
        return 0.5

    probas = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = 0.0, 0.5

    for t in np.linspace(0.05, 0.95, 181):
        preds = (probas >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    return best_t


def evaluate_cumulative(df, project_order, feature_cols):
    rows = []

    for i, held_out in enumerate(project_order):
        prior_projects = project_order[:i]
        test_df = df[df['project'] == held_out]

        if test_df.empty:
            rows.append(_skip_row(i + 1, held_out, 0, 0, 'not in dataset'))
            continue

        if not prior_projects:
            rows.append(_skip_row(i + 1, held_out, 0, 0, 'no training data'))
            continue

        train_df = df[df['project'].isin(prior_projects)]

        if train_df.empty or train_df['label'].sum() == 0:
            rows.append(_skip_row(i + 1, held_out, len(prior_projects), len(train_df), 'no flaky in train'))
            continue

        if test_df['label'].sum() == 0:
            rows.append(_skip_row(i + 1, held_out, len(prior_projects), len(train_df), 'no flaky in test'))
            continue

        X_tr_full = train_df[feature_cols].values
        y_tr_full = train_df['label'].values

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_full, y_tr_full,
                test_size=0.2,
                random_state=42,
                stratify=y_tr_full
            )
            can_tune = len(np.unique(y_val)) == 2
        except ValueError:
            X_tr, y_tr = X_tr_full, y_tr_full
            X_val, y_val = X_tr_full, y_tr_full
            can_tune = False

        if y_tr.sum() >= 2:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model = build_rf()
        model.fit(X_tr, y_tr)

        threshold = find_best_threshold(model, X_val, y_val) if can_tune else 0.5

        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values

        probas = model.predict_proba(X_test)[:, 1]
        y_pred = (probas >= threshold).astype(int)

        rows.append({
            'run_order': i + 1,
            'project': held_out,
            'n_train_proj': len(prior_projects),
            'n_train_tests': len(train_df),
            'n_total': len(y_test),
            'n_flaky': int(y_test.sum()),
            'threshold': round(threshold, 3),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'note': ''
        })

    return pd.DataFrame(rows)


def _skip_row(run_order, project, n_train_proj, n_train_tests, reason):
    return {
        'run_order': run_order,
        'project': project,
        'n_train_proj': n_train_proj,
        'n_train_tests': n_train_tests,
        'n_total': 0,
        'n_flaky': 0,
        'threshold': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'accuracy': None,
        'note': reason
    }

def print_summary(all_results):
    print("\nSUMMARY")

    f1_avgs = []

    for name, df in all_results.items():
        valid = df[df['f1'].notna()]
        if valid.empty:
            continue

        avg_f1 = valid['f1'].mean()
        f1_avgs.append(avg_f1)

        print(f"{name:<30} F1={avg_f1:.4f} (n={len(valid)})")

    if f1_avgs:
        print("\nVariance:")
        print(f"min={min(f1_avgs):.4f} max={max(f1_avgs):.4f} std={np.std(f1_avgs):.6f}")

def main():
    df, feature_cols = load_data(STATIC_FEATURES_CSV)

    permutations = make_permutations(df, n_random=30)

    all_results = {}

    for name, order in permutations.items():
        print(f"\nRunning {name}...")
        results_df = evaluate_cumulative(df, order, feature_cols)
        all_results[name] = results_df

    print_summary(all_results)


if __name__ == "__main__":
    main()