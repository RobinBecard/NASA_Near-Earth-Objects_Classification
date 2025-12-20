from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE


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

    def optimize(self, X_train, y_train, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0, search_method='grid', n_iter=10):
        """
        Hyperparameter search using GridSearchCV or RandomizedSearchCV.

        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Dictionary of hyperparameters to search.
            cv: Number of cross-validation folds.
            scoring: Scoring metric ('roc_auc' by default, falls back to 'f1' if predict_proba not available).
            n_jobs: Number of parallel jobs (-1 uses all cores).
            verbose: Verbosity level.
            search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
            n_iter: Number of iterations for RandomizedSearchCV (ignored for GridSearchCV).

        Returns:
            dict: Best parameters found by the search.
        """
        if self.model is None:
            self._build_model()

        if scoring == 'roc_auc' and not hasattr(self.model, 'predict_proba'):
            print(
                f"Model {self.name} doesn't support predict_proba, using 'f1' instead of 'roc_auc'")
            scoring = 'f1'

        if search_method == 'random':
            from itertools import product
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)

            print(
                f"Building RandomizedSearchCV with n_iter={n_iter}/{total_combinations}, cv={cv}, scoring={scoring}")
            print(f"Training data shape: {X_train.shape}")

            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42
            )
        else:
            print(
                f"Building GridSearchCV with cv={cv}, scoring={scoring}, n_jobs={n_jobs}, verbose={verbose}")
            print(f"Training data shape: {X_train.shape}")

            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )

        search.fit(X_train, y_train)

        print(f"Best score: {search.best_score_:.4f}")

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
        print("  -" * 30)
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")

        if isinstance(metrics['auc_roc'], (int, float)):
            print(f"  - AUC-ROC:   {metrics['auc_roc']:.4f}")
        else:
            print(f"  - AUC-ROC:   {metrics['auc_roc']}")

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
                status = "OK" if pred == true_label else "X"
                result_line = f"    {status} " + \
                    result_line[4:] + f", True={true_label}"

            print(result_line)

        return predictions, probabilities

    def plot_tsne_comparison(self, X_test, y_test, n_samples=None):
        """
        Generate a side-by-side 2D t-SNE visualization.
        Right plot highlights prediction errors in red.

        Args:
            X_test: Test features.
            y_test: True labels.
            n_samples (int): Number of samples to use for t-SNE. If None, use all data.
        """
        # Prepare data
        X_arr = np.array(X_test)
        y_arr = np.array(y_test)

        # Sampling (only if n_samples is specified)
        if n_samples is not None and len(X_arr) > n_samples:
            indices = np.random.choice(len(X_arr), n_samples, replace=False)
            X_sample = X_arr[indices]
            y_sample = y_arr[indices]
        else:
            X_sample = X_arr
            y_sample = y_arr

        # Predict
        y_pred = self.predict(X_sample)

        # Compute t-SNE
        print(f"--> Computing t-SNE on {len(X_sample)} samples...")
        tsne = TSNE(n_components=2, random_state=42,
                    init='random', learning_rate='auto')
        X_embedded = tsne.fit_transform(X_sample)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f't-SNE Visualization - {self.name}', fontsize=16, fontweight='bold')

        # 1. Ground Truth
        sns.scatterplot(
            x=X_embedded[:, 0], y=X_embedded[:, 1],
            hue=y_sample, palette='viridis', ax=axes[0], s=50
        )
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("t-SNE Component 1")
        axes[0].set_ylabel("t-SNE Component 2")

        # 2. Predictions
        sns.scatterplot(
            x=X_embedded[:, 0], y=X_embedded[:, 1],
            hue=y_pred, palette='viridis', ax=axes[1], s=50
        )

        # Highlight errors in red
        errors = (y_sample != y_pred)
        if np.any(errors):
            n_errors = np.sum(errors)
            error_pct = (n_errors / len(y_sample)) * 100
            axes[1].scatter(
                X_embedded[errors, 0], X_embedded[errors, 1],
                facecolors='none', edgecolors='red', s=100, linewidth=1.2, linestyle='-',
                label=f'Errors: {n_errors} ({error_pct:.1f}%)'
            )
            axes[1].legend()

        axes[1].set_title(f"Predictions ({self.name})")
        axes[1].set_xlabel("t-SNE Component 1")
        axes[1].set_ylabel("t-SNE Component 2")

        plt.tight_layout()
        plt.show()
