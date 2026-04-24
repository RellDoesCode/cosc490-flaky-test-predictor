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