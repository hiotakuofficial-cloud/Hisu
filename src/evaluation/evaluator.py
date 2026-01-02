"""Model evaluation"""

import numpy as np
from typing import Dict, Optional, Any
from ..utils.metrics import Metrics


class Evaluator:
    """Evaluate model performance"""

    def __init__(self, model):
        self.model = model
        self.metrics = Metrics()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate model on test set"""
        predictions = self.model.predict(X_test)

        results = {}

        if task_type == 'classification':
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                pred_labels = np.argmax(predictions, axis=1)
            else:
                pred_labels = (predictions > 0.5).astype(int).flatten()

            if y_test.ndim > 1 and y_test.shape[1] > 1:
                true_labels = np.argmax(y_test, axis=1)
            else:
                true_labels = y_test.flatten()

            results['accuracy'] = self.metrics.accuracy(true_labels, pred_labels)
            results['precision'] = self.metrics.precision(true_labels, pred_labels, average='macro')
            results['recall'] = self.metrics.recall(true_labels, pred_labels, average='macro')
            results['f1_score'] = self.metrics.f1_score(true_labels, pred_labels, average='macro')

            cm = self.metrics.confusion_matrix(true_labels, pred_labels)
            results['confusion_matrix'] = cm

        elif task_type == 'regression':
            results['mse'] = self.metrics.mse(y_test, predictions)
            results['rmse'] = self.metrics.rmse(y_test, predictions)
            results['mae'] = self.metrics.mae(y_test, predictions)
            results['r2_score'] = self.metrics.r2_score(y_test, predictions)

        return results

    def evaluate_with_probabilities(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate with probability-based metrics"""
        predictions = self.model.predict(X_test)

        results = {}

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        else:
            pred_labels = (predictions > 0.5).astype(int).flatten()
            true_labels = y_test.flatten()
            results['roc_auc'] = self.metrics.roc_auc_score(true_labels, predictions.flatten())
            results['log_loss'] = self.metrics.log_loss(true_labels, predictions.flatten())

        results['accuracy'] = self.metrics.accuracy(true_labels, pred_labels)
        results['precision'] = self.metrics.precision(true_labels, pred_labels, average='macro')
        results['recall'] = self.metrics.recall(true_labels, pred_labels, average='macro')
        results['f1_score'] = self.metrics.f1_score(true_labels, pred_labels, average='macro')

        return results

    def detailed_report(self, X_test: np.ndarray, y_test: np.ndarray, class_names: Optional[list] = None) -> str:
        """Generate detailed evaluation report"""
        predictions = self.model.predict(X_test)

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        else:
            pred_labels = (predictions > 0.5).astype(int).flatten()
            true_labels = y_test.flatten()

        cm = self.metrics.confusion_matrix(true_labels, pred_labels)
        classes = np.unique(np.concatenate([true_labels, pred_labels]))

        report = "=" * 60 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "Overall Metrics:\n"
        report += f"  Accuracy:  {self.metrics.accuracy(true_labels, pred_labels):.4f}\n"
        report += f"  Precision: {self.metrics.precision(true_labels, pred_labels, average='macro'):.4f}\n"
        report += f"  Recall:    {self.metrics.recall(true_labels, pred_labels, average='macro'):.4f}\n"
        report += f"  F1-Score:  {self.metrics.f1_score(true_labels, pred_labels, average='macro'):.4f}\n\n"

        report += "Per-Class Metrics:\n"
        for i, cls in enumerate(classes):
            cls_name = class_names[i] if class_names and i < len(class_names) else f"Class {cls}"

            cls_mask_true = (true_labels == cls)
            cls_mask_pred = (pred_labels == cls)

            tp = np.sum(cls_mask_true & cls_mask_pred)
            fp = np.sum(~cls_mask_true & cls_mask_pred)
            fn = np.sum(cls_mask_true & ~cls_mask_pred)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            report += f"\n  {cls_name}:\n"
            report += f"    Precision: {precision:.4f}\n"
            report += f"    Recall:    {recall:.4f}\n"
            report += f"    F1-Score:  {f1:.4f}\n"
            report += f"    Support:   {np.sum(cls_mask_true)}\n"

        report += "\n" + "=" * 60 + "\n"

        return report
