import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


STATIC_FEATURES_CSV = "data/flakeflagger/static_features.csv"


# -----------------------------
# FEATURES
# -----------------------------
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
    'imports_mockito', 'imports_powermock', 'imports_easymock',
    'imports_concurrent', 'imports_atomic', 'imports_network',
    'imports_spring', 'imports_guice', 'imports_jdbc', 'imports_jpa',
    'imports_nio', 'imports_io', 'imports_awaitility', 'num_imports',
]


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)

    if 'flaky' in df.columns:
        df = df.rename(columns={'flaky': 'label'})

    df['project'] = df['project'].str.lower().str.strip()
    df['label'] = df['label'].astype(int)

    features = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=features + ['label', 'project'])

    return df, features


# -----------------------------
# MODEL
# -----------------------------
def build_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )


# -----------------------------
# THRESHOLD TUNING
# -----------------------------
def best_threshold(model, X_val, y_val):
    if len(np.unique(y_val)) < 2:
        return 0.5

    probs = model.predict_proba(X_val)[:, 1]

    best_t, best_f1 = 0.5, 0.0

    for t in np.linspace(0.05, 0.95, 100):
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    return best_t


# -----------------------------
# SAFE SMOTE (FIXED)
# -----------------------------
def safe_smote(X_tr, y_tr):
    try:
        class_counts = Counter(y_tr)
        minority_count = min(class_counts.values())

        # must have at least k+1 samples
        k = min(5, minority_count - 1)

        if k >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    except Exception:
        pass

    return X_tr, y_tr


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(df, order, features):
    results = []

    for i in range(1, len(order)):
        train_projects = order[:i]
        test_project = order[i]

        train = df[df.project.isin(train_projects)]
        test = df[df.project == test_project]

        if train.empty or test.empty:
            continue

        X = train[features].values
        y = train['label'].values

        # split train/val
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )
        except ValueError:
            X_tr, y_tr = X, y
            X_val, y_val = X, y

        # SAFE SMOTE
        if len(np.unique(y_tr)) > 1:
            X_tr, y_tr = safe_smote(X_tr, y_tr)

        model = build_xgb()
        model.fit(X_tr, y_tr)

        threshold = best_threshold(model, X_val, y_val)

        X_test = test[features].values
        y_test = test['label'].values

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        results.append({
            "project": test_project,
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "accuracy": accuracy_score(y_test, preds),
        })

    return pd.DataFrame(results)


# -----------------------------
# CURRICULA GENERATION
# -----------------------------
def build_curricula(df):
    stats = df.groupby("project")["label"].mean()
    concurrency = df.groupby("project")["has_concurrency"].mean()
    io = df.groupby("project")["has_network_io"].mean()

    curricula = {
        "flaky_high_to_low": stats.sort_values(ascending=False).index.tolist(),
        "flaky_low_to_high": stats.sort_values().index.tolist(),
        "concurrency_high_to_low": concurrency.sort_values(ascending=False).index.tolist(),
        "io_high_to_low": io.sort_values(ascending=False).index.tolist(),
    }

    # random baselines
    projects = stats.index.tolist()
    for i in range(10):
        curricula[f"random_{i+1}"] = np.random.permutation(projects).tolist()

    return curricula


# -----------------------------
# MAIN
# -----------------------------
def main():
    df, features = load_data(STATIC_FEATURES_CSV)

    print(f"{len(df)} samples loaded | {df.label.sum()} flaky")

    curricula = build_curricula(df)

    all_results = {}

    for name, order in curricula.items():
        print(f"Running {name} ...")
        all_results[name] = evaluate(df, order, features)

    # simple summary
    print("\n=== SUMMARY ===")
    for name, res in all_results.items():
        if res.empty:
            continue
        print(f"{name:<25} F1={res.f1.mean():.4f} (n={len(res)})")


if __name__ == "__main__":
    main()