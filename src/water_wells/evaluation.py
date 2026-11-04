"""Evaluation independent of training and reusable across experiments."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from .data_loader import TARGET_CLASSES


class ModelEvaluator:
    def baseline_predictions(self, X_train, y_train, X_valid,
                             strategy="stratified", random_state=42):
        """Predict validation labels using training class frequencies only.

        stratified draws random labels; uniform guesses equally across classes;
        most_frequent always predicts the most common training label.
        No feature engineering or model pipeline fitting is performed.
        """
        baseline = DummyClassifier(strategy=strategy, random_state=random_state)
        baseline.fit(X_train, y_train)
        return pd.Series(baseline.predict(X_valid), index=X_valid.index,
                         name="baseline_prediction")

    def metrics(self, y_true, y_pred, average="weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=TARGET_CLASSES, average=average, zero_division=0)
        return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision,
                "recall": recall, "f1": f1}

    def classification_report(self, y_true, y_pred, output_dict=False):
        return classification_report(y_true, y_pred, labels=TARGET_CLASSES,
                                     zero_division=0, output_dict=output_dict)

    def confusion_matrix(self, y_true, y_pred):
        return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES),
                            index=TARGET_CLASSES, columns=TARGET_CLASSES)

    def plot_confusion_matrix(self, y_true, y_pred):
        display = ConfusionMatrixDisplay(
            self.confusion_matrix(y_true, y_pred).to_numpy(),
            display_labels=TARGET_CLASSES)
        display.plot(xticks_rotation=30)
        display.ax_.figure.tight_layout()
        return display.ax_

    def feature_importance(self, pipeline):
        return pd.DataFrame({
            "feature": pipeline.preprocessor.get_feature_names_out(),
            "importance": pipeline.classifier.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def plot_feature_importance(self, pipeline, top_n=20):
        data = self.feature_importance(pipeline).head(top_n).iloc[::-1]
        _, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(data["feature"], data["importance"], color="steelblue")
        ax.set(xlabel="XGBoost feature importance", title=f"Top {top_n} features")
        ax.figure.tight_layout()
        return ax
