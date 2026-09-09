"""Train, evaluate, and save churn prediction models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def set_seed() -> None:
    """Make model training as reproducible as possible."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load Telco data and apply only verified cleaning operations."""
    data = pd.read_csv(path)
    duplicate_count = data.duplicated().sum()
    if duplicate_count:
        data = data.drop_duplicates().copy()

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # Blank TotalCharges occurs for new customers with tenure 0. Median imputation
    # preserves those records instead of discarding customers without charge history.
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    if data["Churn"].isna().any():
        raise ValueError("Churn contains unexpected labels.")
    return data


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create a preprocessing transformer without fitting it."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def metrics_from_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate all reported binary-classification metrics at a 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
    }


def save_evaluation_artifacts(
    name: str, y_test: pd.Series, probabilities: np.ndarray
) -> dict[str, float]:
    """Save confusion matrix, ROC curve, and text report for a model."""
    metrics = metrics_from_probabilities(y_test, probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    REPORTS_DIR.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["No Churn", "Churn"]).plot(ax=axis, colorbar=False)
    axis.set_title(f"{name} Confusion Matrix")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{name} ROC Curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png", dpi=160)
    plt.close(figure)

    (REPORTS_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["No Churn", "Churn"]), encoding="utf-8"
    )
    return metrics


def build_dnn(input_size: int, learning_rate: float = 0.001) -> Sequential:
    """Build a compact feedforward neural network for transformed features."""
    model = Sequential([
        tf.keras.Input(shape=(input_size,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Save DNN training and validation curves."""
    REPORTS_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="DNN Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="DNN Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "dnn_training_history.png", dpi=160)
    plt.close(figure)


def main() -> None:
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    data = load_and_clean_data(DATA_PATH)
    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, numeric_features, categorical_features = make_preprocessor(X)
    # The transformer is fitted only to training data, preventing test-data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    logistic_search = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 1.0, 10.0]}, scoring="roc_auc", cv=5, n_jobs=-1
    )
    logistic_search.fit(X_train_processed, y_train)
    logistic_model = logistic_search.best_estimator_
    joblib.dump(logistic_model, MODELS_DIR / "logistic_regression.pkl")
    logistic_probabilities = logistic_model.predict_proba(X_test_processed)[:, 1]
    logistic_metrics = save_evaluation_artifacts("Logistic Regression", y_test, logistic_probabilities)

    dnn_model = build_dnn(X_train_processed.shape[1])
    callbacks = [EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
    history = dnn_model.fit(
        X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    dnn_model.save(MODELS_DIR / "churn_dnn.keras")
    X_test_dnn = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
    dnn_probabilities = dnn_model.predict(X_test_dnn, verbose=0).ravel()
    dnn_metrics = save_evaluation_artifacts("Deep Neural Network", y_test, dnn_probabilities)
    plot_history(history)

    comparison = pd.DataFrame([
        {"Model": "Logistic Regression", **logistic_metrics},
        {"Model": "Deep Neural Network", **dnn_metrics},
    ])
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_model = comparison.loc[comparison["roc_auc"].idxmax(), "Model"]
    metadata = {
        "best_model_by_roc_auc": best_model,
        "logistic_best_C": logistic_search.best_params_["C"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "rows_after_cleaning": len(data),
        "test_rows": len(y_test),
    }
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nSelected final model by ROC-AUC: {best_model}")


if __name__ == "__main__":
    main()
