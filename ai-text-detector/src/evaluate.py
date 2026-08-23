from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def evaluate(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)

    # SGDClassifier doesn't have predict_proba by default so gotta handle that
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def print_report(name, metrics):
    print(name)
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  roc_auc:   {metrics['roc_auc']:.4f}")
    print(f"  confusion matrix: {metrics['confusion_matrix']}")
