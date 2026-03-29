from src.dataset_loader import load_dataset_from_csv
from src.data_cleaning import clean_dataset
from src.stats import dataset_stats, check_imbalance, print_sample
from src.feature_extractor import prepare_features
from src.models import train_random_forest, train_xgboost
from src.evaluation import evaluate_model, detailed_evaluation
# from src.shap_analysis import run_shap_analysis  # optional (needs shap installed)

import pandas as pd


def main():
    file_path = "data/flakeflagger/processed_data.csv"

    print("Loading dataset...")
    data = load_dataset_from_csv(file_path)
    print(f"Loaded {len(data)} rows")

    # Cleaning
    data = clean_dataset(data)

    # Stats
    dataset_stats(data)
    check_imbalance(data)
    print_sample(data)

    # Save cleaned dataset
    df = pd.DataFrame(data)
    df.to_excel("cleaned_dataset.xlsx", index=False)

    # Feature extraction
    X, y = prepare_features(data, drop_runtime=True)

    # Train models
    rf_model = train_random_forest(X, y)
    xgb_model = train_xgboost(X, y)

    # Evaluate
    print("\nRandom Forest Results:")
    evaluate_model(rf_model, X, y)
    detailed_evaluation(rf_model, X, y)

    print("\nXGBoost Results:")
    evaluate_model(xgb_model, X, y)
    detailed_evaluation(xgb_model, X, y)

    # Optional SHAP (uncomment when ready)
    # run_shap_analysis(rf_model, X)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()