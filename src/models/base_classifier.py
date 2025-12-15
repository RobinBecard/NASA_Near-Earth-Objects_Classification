from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV


class BaseClassifier(ABC):
    """
    Abstract base class for classification models.
    This class cannot be used directly. It serves as a template
    for concrete model implementations.
    """

    def __init__(self, name, params):
        """
        Initialize the base model.
        Args:
            name (str): Name of the model.
            params (dict): Dictionary of model hyperparameters.
        """
        self.name = name
        self.params = params
        self.model = None  # The sklearn model will be stored here

    @abstractmethod
    def _build_model(self):
        """Build the model instance. Must be implemented by subclasses."""
        pass

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        print(f"--> Training {self.name}...")
        if self.model is None:
            self._build_model()
        self.model.fit(X_train, y_train)

    def optimize(self, X_train, y_train, param_grid):
        """Hyperparameter search using GridSearchCV."""
        search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the provided test data.
        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        print(f"--> Evaluating {self.name}...")
        predictions = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        # 2. Probability predictions (For AUC-ROC)
        # Note: We take column [:, 1] which corresponds to the probability of the positive class (1)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probas)
        else:
            # Some models (like SVM without config) do not provide probabilities
            auc = "N/A"
        return {
            "model_name": self.name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "auc_roc": auc,
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        }

    def display_metrics(self, metrics: dict) -> None:
        """
        Display formatted evaluation metrics.

        Args:
            metrics (dict): Dictionary of metrics from evaluate().
        """
        print(f"\nRESULTS - {metrics['model_name']}:")
        print("  ─" * 30)
        print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall:    {metrics['recall']:.4f}")
        print(f"  • F1-Score:  {metrics['f1_score']:.4f}")

        if isinstance(metrics['auc_roc'], (int, float)):
            print(f"  • AUC-ROC:   {metrics['auc_roc']:.4f}")
        else:
            print(f"  • AUC-ROC:   {metrics['auc_roc']}")

        print("\n  Confusion matrix:")
        cm = metrics['confusion_matrix']
        print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
        print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")

    def predict_with_details(self, X, y_true=None, n_samples: int = 5):
        """
        Make predictions and display detailed results.

        Args:
            X: Features to predict on.
            y_true: True labels (optional, for comparison).
            n_samples (int): Number of samples to display.

        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = self.predict(X)
        probabilities = self.model.predict_proba(X) if hasattr(
            self.model, "predict_proba") else None

        print(
            f"\n  Detailed predictions ({min(n_samples, len(predictions))} samples):")
        for i in range(min(n_samples, len(predictions))):
            pred = predictions[i]

            result_line = f"    Sample {i+1}: Predicted={pred}"

            if probabilities is not None:
                proba = probabilities[i][1]  # Positive class probability
                result_line += f", Proba={proba:.3f}"

            if y_true is not None:
                true_label = y_true.iloc[i] if hasattr(
                    y_true, 'iloc') else y_true[i]
                status = "✓" if pred == true_label else "✗"
                result_line = f"    {status} " + \
                    result_line[4:] + f", True={true_label}"

            print(result_line)

        return predictions, probabilities
