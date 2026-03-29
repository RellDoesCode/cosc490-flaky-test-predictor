from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_random_forest(X, y):
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X, y)
    return model


def train_xgboost(X, y):
    print("\nTraining XGBoost...")

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X, y)
    return model