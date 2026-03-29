from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


def evaluate_model(model, X, y):
    print("\n--- Cross-Validation ---")

    scores = cross_val_score(model, X, y, cv=5, scoring='f1')

    print("F1 scores:", scores)
    print("Average F1:", scores.mean())


def detailed_evaluation(model, X, y):
    print("\n--- Detailed Evaluation ---")

    y_pred = model.predict(X)
    print(classification_report(y, y_pred))