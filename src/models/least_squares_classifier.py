from sklearn.linear_model import RidgeClassifier
from src.models.base_classifier import BaseModel
from src.config import get_config
import matplotlib.pyplot as plt
import numpy as np

class LeastSquaresClassifier(BaseModel):
    """
    Implementation of the Least Squares Classifier (Ridge Classifier).
    It treats classification as a regression problem by minimizing 
    the Mean Squared Error (MSE) on targets -1 and +1.
    """

    def _build_model(self):
        """
        Builds the RidgeClassifier model with hyperparameters from config.yaml.
        
        Hyperparameters:
            - alpha: Regularization strength (Higher values = stronger regularization/less overfitting).
            - class_weight: Weights associated with classes to handle imbalance.
            - solver: Solver to use in the computational routines ('auto', 'svd', 'cholesky', etc.).
        """
        config = get_config()
        default_params = config.get_param('models.least_squares', {})
        
        self.model = RidgeClassifier(
            alpha=self.params.get('alpha', default_params.get('alpha', 1.0)),
            class_weight=self.params.get('class_weight', default_params.get('class_weight', None)),
            solver=self.params.get('solver', default_params.get('solver', 'auto')),
            random_state=self.params.get('random_state', default_params.get('random_state', 42))
        )

    def plot_feature_importance(self, feature_names=None):
        """
        Visualizes the model coefficients (weights).
        Shows which features contribute most to the decision (+ or -).
        """
        if self.model is None:
            print("Error: Model not trained.")
            return

        # Coefficients are stored in self.model.coef_
        # For binary classification, shape is (1, n_features)
        if hasattr(self.model, 'coef_'):
            importances = self.model.coef_[0]
        else:
            print("Error: Coefficients not available.")
            return

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        indices = np.argsort(np.abs(importances))
        
        sorted_importances = importances[indices]
        sorted_names = np.array(feature_names)[indices]
        
        colors = ['#ff9999' if x < 0 else '#99ff99' for x in sorted_importances]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importances)), sorted_importances, color=colors, edgecolor='k')
        plt.yticks(range(len(sorted_importances)), sorted_names)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.8)
        
        plt.xlabel("Coefficient Value (Least Squares Weight)")
        plt.title(f"Feature Importance - {self.name} (MSE Optimization)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()