import shap

def run_shap_analysis(model, X):
    print("\nRunning SHAP analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)