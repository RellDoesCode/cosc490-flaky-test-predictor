import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok = True)

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

FEATURE_LABELS = [
    'Lines of Code', 'Assert Count', 'Thread.sleep Count', 'Has Thread.sleep',
    'Async Wait Count', 'Has Async Wait', 'Has File I/O', 'Has Network I/O',
    'Has Concurrency', 'Num Test Methods', 'Num Try/Catch',
    'Has Setup/Teardown', 'Num Conditionals', 'Has Random',
    'Has System Time', 'Num Annotations',
    'Assert Density', 'LOC per Test', 'Has @Test(timeout)', 'Timeout Count',
    'Polling Count', 'Has Env Access', 'Has DB Access', 'Has Injection',
    'Has Static Field', 'Thread Join Count', 'Notify Count', 'Broad Catch Count',
    'File I/O Count', 'Network I/O Count', 'Has @Rule', 'Num Inner Classes',
    'Imports Mockito', 'Imports PowerMock', 'Imports EasyMock',
    'Imports Concurrent', 'Imports Atomic', 'Imports Network',
    'Imports Spring', 'Imports Guice', 'Imports JDBC', 'Imports JPA',
    'Imports NIO', 'Imports IO', 'Imports Awaitility', 'Num Imports',
]

def _get_shap_values_for_positive_class(explainer, X_array):
    raw = explainer.shap_values(X_array)
    if isinstance(raw, list):
        return raw[1]
    else:
        return raw

def run_shap_analysis(model, X, feature_names = None, model_label = "Model", max_display = 20):
    print(f"\n{'='*60}")
    print(f" SHAP Analysis - {model_label}")
    print(f"{'='*60}")

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        elif len(FEATURE_LABELS) == X.shape[1]:
            feature_names = FEATURE_LABELS
        elif len(FEATURE_COLS) == X.shape[1]:
            feature_names = FEATURE_COLS
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    X_array = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_matrix = _get_shap_values_for_positive_class(explainer, X_array)

    mean_abs_shap = np.abs(shap_matrix).mean(axis = 0)
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending = False).reset_index(drop = True)
    ranking_df['rank'] = ranking_df.index + 1

    print(f"\nTop {min(max_display, len(ranking_df))} most important features:\n")
    print(f" {'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}")
    print(f" {'-'*5} {'-'*30} {'-'*12}")
    for _, row in ranking_df.head(max_display).iterrows():
        print(f" {int(row['rank']):<5} {row['feature']:<30} {row['mean_abs_shap']:.4f}")

    safe_label = model_label.lower().replace(" ", "_")
    csv_path = os.path.join(RESULTS_DIR, f"shap_feature_ranking_{safe_label}.csv")
    ranking_df.to_csv(csv_path, index = False)
    print(f"\nFeature ranking saved to: {csv_path}")