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

TARGET_PROJECTS = [
    'achilles', 'activiti', 'ambari', 'assertj-core', 'commons-exec',
    'handlebars.java', 'hbase', 'hector', 'incubator-dubbo', 'java-websocket',
    'logback', 'ninja', 'spring-boot', 'undertow', 'wildfly', 'wro4j', 'zxing',
]

PERMUTATIONS = {
    "Permutation 1 — Alphabetical (baseline)": [
        'achilles', 'activiti', 'ambari', 'assertj-core', 'commons-exec',
        'handlebars.java', 'hbase', 'hector', 'incubator-dubbo', 'java-websocket',
        'logback', 'ninja', 'spring-boot', 'undertow', 'wildfly', 'wro4j', 'zxing',
    ],
    "Permutation 2 — Reversed": [
        'zxing', 'wro4j', 'wildfly', 'undertow', 'spring-boot', 'ninja', 'logback',
        'java-websocket', 'incubator-dubbo', 'hector', 'hbase', 'handlebars.java',
        'commons-exec', 'assertj-core', 'ambari', 'activiti', 'achilles',
    ],
    "Permutation 3 — Large projects first": [
        'spring-boot', 'wildfly', 'ambari', 'hbase', 'activiti', 'incubator-dubbo',
        'undertow', 'logback', 'assertj-core', 'zxing', 'ninja', 'wro4j',
        'handlebars.java', 'hector', 'achilles', 'java-websocket', 'commons-exec',
    ],
    "Permutation 4 — Small / unit-test projects first": [
        'commons-exec', 'java-websocket', 'hector', 'achilles', 'handlebars.java',
        'wro4j', 'ninja', 'zxing', 'assertj-core', 'logback', 'incubator-dubbo',
        'undertow', 'hbase', 'activiti', 'ambari', 'wildfly', 'spring-boot',
    ],
}

def load_data(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    if 'flaky' in df.columns:
        df = df.rename(columns={'flaky': 'label'})

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} feature(s) not in CSV, skipping: {missing}\n")

    df = df.dropna(subset=available_features + ['label', 'project'])
    df['label'] = df['label'].astype(int)
    df['project'] = df['project'].str.lower().str.strip()
    return df, available_features

def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )

def find_best_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Sweep thresholds on a validation set and return the one maximising F1."""
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

def evaluate_cumulative(
    df: pd.DataFrame,
    project_order: list,
    feature_cols: list,
) -> pd.DataFrame:

    rows = []

    for i, held_out in enumerate(project_order):
        prior_projects = project_order[:i]
        test_df = df[df['project'] == held_out]

        if test_df.empty:
            rows.append(_skip_row(i + 1, held_out, 0, 0, 'not in dataset'))
            continue

        if not prior_projects:
            rows.append(_skip_row(i + 1, held_out, 0, 0, 'first in order — no training data yet'))
            continue

        train_df = df[df['project'].isin(prior_projects)]

        if train_df.empty or train_df['label'].sum() == 0:
            rows.append(_skip_row(i + 1, held_out, len(prior_projects), len(train_df),
                                  'train set has no flaky examples'))
            continue

        if test_df['label'].sum() == 0:
            rows.append(_skip_row(i + 1, held_out, len(prior_projects), len(train_df),
                                  'no flaky tests in test set'))
            continue

        X_tr_full = train_df[feature_cols].values
        y_tr_full = train_df['label'].values

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_full, y_tr_full,
                test_size=0.20, random_state=42, stratify=y_tr_full
            )
            can_tune = len(np.unique(y_val)) == 2
        except ValueError:
            X_tr, y_tr = X_tr_full, y_tr_full
            X_val, y_val = X_tr_full, y_tr_full
            can_tune = False

        # this applies SMOTE to training split only (not validation or test)
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
            'run_order':     i + 1,
            'project':       held_out,
            'n_train_proj':  len(prior_projects),
            'n_train_tests': len(train_df),
            'n_total':       len(y_test),
            'n_flaky':       int(y_test.sum()),
            'threshold':     round(threshold, 3),
            'precision':     round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':        round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1':            round(f1_score(y_test, y_pred, zero_division=0), 4),
            'accuracy':      round(accuracy_score(y_test, y_pred), 4),
            'note':          '',
        })

    return pd.DataFrame(rows)

def _skip_row(run_order, project, n_train_proj, n_train_tests, reason):
    return {
        'run_order': run_order, 'project': project,
        'n_train_proj': n_train_proj, 'n_train_tests': n_train_tests,
        'n_total': 0, 'n_flaky': 0,
        'threshold': None, 'precision': None,
        'recall': None, 'f1': None, 'accuracy': None,
        'note': reason,
    }

W = dict(ord=4, proj=20, ntp=5, ntt=7, ntot=6, nfl=5,
         thr=6, prec=10, rec=8, f1=8, acc=9)

SEP = "=" * 106
DASH = "-" * 106

def _header():
    return (
        f"{'#':>{W['ord']}}  {'Project':<{W['proj']}}  "
        f"{'TrPr':>{W['ntp']}}  {'TrN':>{W['ntt']}}  "
        f"{'N':>{W['ntot']}}  {'Flky':>{W['nfl']}}  "
        f"{'Thr':>{W['thr']}}  {'Precision':>{W['prec']}}  "
        f"{'Recall':>{W['rec']}}  {'F1':>{W['f1']}}  "
        f"{'Accuracy':>{W['acc']}}  Note"
    )

def _row_str(row):
    if row['f1'] is None:
        thr  = f"{'—':>{W['thr']}}"
        vals = f"  {'—':>{W['prec']}}  {'—':>{W['rec']}}  {'—':>{W['f1']}}  {'—':>{W['acc']}}"
    else:
        thr  = f"{row['threshold']:>{W['thr']}.3f}"
        vals = (
            f"  {row['precision']:>{W['prec']}.4f}  "
            f"{row['recall']:>{W['rec']}.4f}  "
            f"{row['f1']:>{W['f1']}.4f}  "
            f"{row['accuracy']:>{W['acc']}.4f}"
        )
    return (
        f"{int(row['run_order']):>{W['ord']}}  "
        f"{row['project']:<{W['proj']}}  "
        f"{int(row['n_train_proj']):>{W['ntp']}}  "
        f"{int(row['n_train_tests']):>{W['ntt']}}  "
        f"{int(row['n_total']):>{W['ntot']}}  "
        f"{int(row['n_flaky']):>{W['nfl']}}  "
        f"{thr}{vals}  {row['note']}"
    )

def print_permutation_results(perm_name, results_df):
    valid = results_df[results_df['f1'].notna()]

    print(f"\n{SEP}")
    print(f"  {perm_name}")
    print(f"  TrPr = # training projects seen before this one | TrN = # training tests")
    print(SEP)
    print(_header())
    print(DASH)

    for _, row in results_df.iterrows():
        print(_row_str(row))

    if not valid.empty:
        print(DASH)
        print(
            f"{'AVG':>{W['ord']}}  {'(macro avg)':<{W['proj']}}  "
            f"{'':>{W['ntp']}}  {'':>{W['ntt']}}  "
            f"{'':>{W['ntot']}}  {'':>{W['nfl']}}  "
            f"{'':>{W['thr']}}  "
            f"{valid['precision'].mean():>{W['prec']}.4f}  "
            f"{valid['recall'].mean():>{W['rec']}.4f}  "
            f"{valid['f1'].mean():>{W['f1']}.4f}  "
            f"{valid['accuracy'].mean():>{W['acc']}.4f}"
        )

def print_delta_table(all_results):
    """Per-project F1 delta vs. Permutation 1 (baseline)."""
    names = list(all_results.keys())
    baseline_df = all_results[names[0]].set_index('project')


    col_labels = [n.split("—")[0].strip() for n in names[1:]]
    header = f"  {'Project':<22}" + "".join(f"  {c:<16}" for c in col_labels)
    print(header)
    print("  " + "-" * (22 + 18 * len(col_labels)))

    all_projects = sorted({
        p for df in all_results.values()
        for p in df[df['f1'].notna()]['project'].tolist()
    })

    for proj in all_projects:
        base_row = baseline_df.loc[proj] if proj in baseline_df.index else None
        base_f1  = base_row['f1'] if base_row is not None and pd.notna(base_row['f1']) else None

        line = f"  {proj:<22}"
        for perm_name in names[1:]:
            perm_df  = all_results[perm_name].set_index('project')
            perm_row = perm_df.loc[proj] if proj in perm_df.index else None
            perm_f1  = perm_row['f1'] if perm_row is not None and pd.notna(perm_row['f1']) else None

            if base_f1 is None or perm_f1 is None:
                cell = "     —      "
            else:
                delta = perm_f1 - base_f1
                cell  = f"  {delta:+.4f}      "
            line += f"  {cell:<16}"
        print(line)

def print_summary(all_results):
    print(f"\n{SEP}")
    print("  CROSS-PERMUTATION SUMMARY  (macro-averages over evaluated projects)")
    print(SEP)
    print(f"  {'Permutation':<45}  {'Projects':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Accuracy':>9}")
    print("  " + "-" * 96)

    f1_avgs, prec_avgs, rec_avgs = [], [], []

    for perm_name, df in all_results.items():
        valid = df[df['f1'].notna()]
        if valid.empty:
            continue
        short = (perm_name.split("—")[0].strip() + " — " + perm_name.split("—")[1].strip()
                 if "—" in perm_name else perm_name)
        avg_f1   = valid['f1'].mean()
        avg_prec = valid['precision'].mean()
        avg_rec  = valid['recall'].mean()
        avg_acc  = valid['accuracy'].mean()
        f1_avgs.append(avg_f1)
        prec_avgs.append(avg_prec)
        rec_avgs.append(avg_rec)
        print(f"  {short:<45}  {len(valid):>8}  {avg_prec:>10.4f}  "
              f"{avg_rec:>8.4f}  {avg_f1:>8.4f}  {avg_acc:>9.4f}")

    if not f1_avgs:
        return

    print(f"\n  Variance across permutations:")
    print(f"    F1        — min={min(f1_avgs):.4f}  max={max(f1_avgs):.4f}  "
          f"range={max(f1_avgs)-min(f1_avgs):.4f}  std={np.std(f1_avgs):.6f}")
    print(f"    Precision — min={min(prec_avgs):.4f}  max={max(prec_avgs):.4f}  "
          f"range={max(prec_avgs)-min(prec_avgs):.4f}  std={np.std(prec_avgs):.6f}")
    print(f"    Recall    — min={min(rec_avgs):.4f}  max={max(rec_avgs):.4f}  "
          f"range={max(rec_avgs)-min(rec_avgs):.4f}  std={np.std(rec_avgs):.6f}")

def main():
    df, feature_cols = load_data(STATIC_FEATURES_CSV)

    all_results = {}

    for perm_name, project_order in PERMUTATIONS.items():
        print(f"\nRunning {perm_name}...")
        results_df = evaluate_cumulative(df, project_order, feature_cols)
        print_permutation_results(perm_name, results_df)
        all_results[perm_name] = results_df

    print_summary(all_results)
    print_delta_table(all_results)

if __name__ == "__main__":
    main()
