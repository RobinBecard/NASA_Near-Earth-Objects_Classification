from sklearn.linear_model import LogisticRegression
from src.models.base_classifier import BaseModel
from src.config import get_config
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressionClassifier(BaseModel):
    """
    Specific implementation for Logistic Regression.
    Inherits from BaseModel.
    """

    def _build_model(self):
        """
        Build the LogisticRegression model with hyperparameters from config.yaml.
        
        Hyperparameters:
            - penalty: Regularization term ('l1', 'l2', 'elasticnet', 'none').
            - C: Inverse of regularization strength (smaller values specify stronger regularization).
            - solver: Algorithm to use in the optimization problem ('liblinear', 'lbfgs', etc.).
        """
        config = get_config()
        default_params = config.get_param('models.logistic_regression', {})
        
        self.model = LogisticRegression(
            penalty=self.params.get('penalty', default_params.get('penalty', 'l2')),
            C=self.params.get('C', default_params.get('C', 1.0)),
            solver=self.params.get('solver', default_params.get('solver', 'lbfgs')),
            max_iter=self.params.get('max_iter', default_params.get('max_iter', 100)),
            class_weight=self.params.get('class_weight', default_params.get('class_weight', None)),
            random_state=self.params.get('random_state', default_params.get('random_state', 42))
        )

    def plot_feature_importance(self, feature_names=None):
        """
        Visualizes the coefficients (weights) of the model.
        This shows how much each feature contributes to the positive class.
        
        Args:
            feature_names (list): List of feature names. If None, indices are used.
        """
        if self.model is None:
            print("Error: Model not trained.")
            return

        # Get coefficients (weights)
        # For binary classification, shape is (1, n_features)
        if hasattr(self.model, 'coef_'):
            importances = self.model.coef_[0]
        else:
            print("Error: Model has no coefficients available.")
            return

        # Create generic names if none provided
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
        
        plt.xlabel("Coefficient Value (Log Odds)")
        plt.title(f"Feature Importance (Coefficients) - {self.name}")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()