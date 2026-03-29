import pandas as pd

def load_dataset_from_csv(file_path, drop_runtime=False):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        print("Standard read failed, trying fallback encoding...")
        df = pd.read_csv(file_path, encoding='latin1')

    # Rename label column
    if 'flaky' in df.columns:
        df = df.rename(columns={'flaky': 'label'})
    else:
        raise ValueError("Dataset must contain a 'flaky' column")

    # Optionally drop runtime-heavy features
    if drop_runtime:
        runtime_cols = [
            'ExecutionTime',
            'numCoveredLines',
            'numCoveredMethods'
        ]
        df = df.drop(columns=runtime_cols, errors='ignore')

    return df.to_dict(orient='records')