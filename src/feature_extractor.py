import pandas as pd

def prepare_features(data, drop_runtime=False):
    df = pd.DataFrame(data)

    # Separate labels
    y = df['label']

    # Drop known non-feature columns
    drop_cols = ['label', 'test_name', 'project', 'module', 'class', 'Unnamed: 0']

    if drop_runtime:
        drop_cols += ['ExecutionTime', 'numCoveredLines', 'numCoveredMethods']

    df = df.drop(columns=drop_cols, errors='ignore')

    # Keep ONLY numeric columns
    X = df.select_dtypes(include=['number'])

    print("\nFinal feature columns:")
    print(list(X.columns))

    print(f"\nDropped {df.shape[1] - X.shape[1]} non-numeric columns")

    return X, y