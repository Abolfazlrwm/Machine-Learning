import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates trained models using multiple classification metrics.
    Returns a clean comparison table.
    """
    results = []

    for name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = y_pred # Fallback for models without proba

        # Calculate metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_probs)
        }
        results.append(metrics)

    # Create a clean DataFrame to compare models
    comparison_df = pd.DataFrame(results).set_index("Model")
    return comparison_df

def get_feature_importance(model, feature_names):
    """
    Extracts and returns feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)
    return None