from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report


def evaluate_model(model, X, y):
    print("\n--- Stratified 5-Fold Cross-Validation ---")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

    print("F1 scores:", scores)
    print("Average F1:", scores.mean())


def detailed_evaluation(model, X, y):
    print("\n--- Detailed Evaluation (cross-validated predictions) ---")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    print(classification_report(y, y_pred))