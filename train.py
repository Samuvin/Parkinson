"""Complete training pipeline for Parkinson's Disease Prediction System."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import ModelEvaluator
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_model import SVMModel
from src.utils.config import Config, get_models_dir


@dataclass
class VoiceTrainingResult:
    """Outputs from the tabular voice training pipeline (LR + SVM on UCI-style features)."""

    feature_names: List[str]
    X_test: np.ndarray
    y_test: np.ndarray
    lr_metrics: Dict[str, float]
    svm_metrics: Dict[str, float]
    y_pred_lr: np.ndarray
    y_proba_lr: np.ndarray
    y_pred_svm: np.ndarray
    y_proba_svm: np.ndarray
    lr_model: LogisticRegressionModel
    svm_model: SVMModel
    best_model_name: str
    preprocessor: DataPreprocessor


def run_voice_training(
    config: Optional[Config] = None,
    *,
    save_preprocessed: bool = True,
) -> VoiceTrainingResult:
    """
    Load speech tabular data, preprocess, train Logistic Regression and SVM, save models.

    This is the same pipeline as ``main()`` but returns test-set predictions and metrics
    for reporting and downstream tooling (e.g. ``scripts/train_voice_pipeline.py``).

    Args:
        config: Optional ``Config`` instance (default: load ``config.yaml``).
        save_preprocessed: If True, persist processed arrays and scaler under
            ``data/processed/`` and ``models/scaler.joblib`` (same as CLI training).

    Returns:
        VoiceTrainingResult with models, test features/labels, predictions, and metrics.
    """
    config = config or Config()

    print("Step 1: Loading speech (voice) dataset...")
    print("-" * 80)
    loader = DataLoader(config)
    X, y = loader.load_speech_data()
    print(f"\n✓ Speech data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("  Training speech model for multi-model ensemble\n")
    feature_names = list(X.columns)

    print("Step 2: Preprocessing data...")
    print("-" * 80)
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X,
        y,
        remove_outliers=False,
        balance_classes=True,
        save=save_preprocessed,
    )

    print("\n\nStep 3: Training Logistic Regression Baseline...")
    print("-" * 80)
    lr_model = LogisticRegressionModel(config)
    lr_model.train(X_train, y_train, X_val, y_val)
    lr_test_metrics = lr_model.evaluate(X_test, y_test)
    lr_model.save_model("logistic_regression_model.joblib")

    print("\n\nStep 4: Training SVM with Kernel Optimization...")
    print("-" * 80)
    svm_model = SVMModel(config)
    svm_model.train(X_train, y_train, X_val, y_val, search_method="grid")
    svm_model.print_kernel_comparison()
    svm_test_metrics = svm_model.evaluate(X_test, y_test)
    svm_model.save_model("svm_model.joblib")

    print("\n\nStep 5: Model Comparison...")
    print("-" * 80)
    evaluator = ModelEvaluator(config)
    comparison = evaluator.compare_models(
        {
            "Logistic Regression": lr_test_metrics,
            "SVM (Optimized)": svm_test_metrics,
        },
        save_path=get_models_dir() / "model_comparison.png",
    )
    print("\nModel Comparison Results:")
    print(comparison.to_string())

    if svm_test_metrics["accuracy"] > lr_test_metrics["accuracy"]:
        print("\n\nSVM achieved better accuracy. Saving as best_model.joblib")
        svm_model.save_model("best_model.joblib")
        best_model_name = "SVM"
        best_metrics = svm_test_metrics
    else:
        print("\n\nLogistic Regression achieved better accuracy. Saving as best_model.joblib")
        lr_model.save_model("best_model.joblib")
        best_model_name = "Logistic Regression"
        best_metrics = lr_test_metrics

    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    y_pred_svm = svm_model.predict(X_test)
    y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

    print("\n\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:    {best_metrics['roc_auc']:.4f}")
    print("\nModels saved to: " + str(get_models_dir()))
    print("Preprocessed data saved to: data/processed/")
    print("\nYou can now run the web application:")
    print("  python webapp/app.py")
    print("\n" + "=" * 80 + "\n")

    return VoiceTrainingResult(
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        lr_metrics=lr_test_metrics,
        svm_metrics=svm_test_metrics,
        y_pred_lr=y_pred_lr,
        y_proba_lr=y_proba_lr,
        y_pred_svm=y_pred_svm,
        y_proba_svm=y_proba_svm,
        lr_model=lr_model,
        svm_model=svm_model,
        best_model_name=best_model_name,
        preprocessor=preprocessor,
    )


def main():
    """Main training pipeline (CLI entrypoint)."""
    print("\n" + "=" * 80)
    print("PARKINSON'S DISEASE PREDICTION SYSTEM - TRAINING PIPELINE")
    print("=" * 80 + "\n")

    config = Config()
    print("Configuration loaded successfully!\n")
    run_voice_training(config=config, save_preprocessed=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
