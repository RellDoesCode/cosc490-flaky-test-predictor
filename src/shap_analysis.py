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

    top_n = min(max_display, len(ranking_df))
    top_df = ranking_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(
        top_df['feature'],
        top_df['mean_abs_shap'],
        color='#3a7abf',
        edgecolor='white',
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=11)
    ax.set_title(
        f"{model_label} - Top {top_n} Features by SHAP Importance\n"
        f"(higher = stronger predictor of flakiness)",
        fontsize=12, pad=14,
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()

    bar_path = os.path.join(RESULTS_DIR, f"shap_summary_bar_{safe_label}.png")
    fig.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Bar chart saved to: {bar_path}")

    explanation = shap.Explanation(
        values = shap_matrix,
        base_values = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
        data = X_array,
        feature_names = feature_names,
    )

    fig2, ax2 = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    shap.plots.beeswarm(
        explanation,
        max_display = top_n,
        show = False,
        plot_size = None,
    )
    plt.title(
        f"{model_label} - SHAP Beeswarm Plot\n"
        f"Red = feature value high | Blue = feature value low\n"
        f"Right of center = pushes toward FLAKy prediction",
        fontsize = 11, pad = 12,
    )
    plt.tight_layout()

    bee_path = os.path.join(RESULTS_DIR, f"shap_beeswarm_{safe_label}.png")
    plt.savefig(bee_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Beeswarm plot saved to: {bee_path}")

    print(f"\nSHAP analysis complete for {model_label}.")
    return ranking_df

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    STATIC_CSV = "data/flakeflagger/static_features.csv"

    print(f"Loading data from {STATIC_CSV}...")
    df = pd.read_csv(STATIC_CSV)

    required = FEATURE_COLS + ['label']
    df = df.dropna(subset = required)
    df['label'] = df['label'].astype(int)

    X = df[FEATURE_COLS].values
    y = df['label'].values
    
    print(f"Dataset: {len(df)} tests | flaky: {y.sum()} ({y.mean() * 100:.1f}%)")

    print("\nTraining Random Forest for SHAP...")
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight = 'balanced')
    rf.fit(X, y)
    run_shap_analysis(rf, X, feature_names = FEATURE_LABELS, model_label = "Random Forest")

    print("\nTraining XGBoost for SHAP...")
    xgb = XGBClassifier(eval_metric = 'logloss', verbosity = 0)
    xgb.fit(X, y)
    run_shap_analysis(xgb, X, feature_names = FEATURE_LABELS, model_label = "XGBoost")

    print("\nAll SHAP outputs saved to results/")