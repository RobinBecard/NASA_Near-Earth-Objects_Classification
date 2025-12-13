from sklearn.svm import SVC
from src.models.base_classifier import BaseModel
from src.config import get_config
import numpy as np
import matplotlib.pyplot as plt

class SVMClassifier(BaseModel):
    """
    Specific implementation for Support Vector Machine (SVM).
    Inherits from BaseModel.
    """

    def _build_model(self):
        """
        Build the SVM model with hyperparameters from config.yaml.

        Hyperparameters:
            - kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            - C: Regularization parameter (must be strictly positive)
            - probability: Whether to enable probability estimates (Required for AUC)
            - class_weight: Set to 'balanced' to handle imbalanced datasets
            - random_state: Random state for reproducibility
        """
        config = get_config()
        
        # Get default hyperparameters from config if not provided in self.params
        default_params = config.get_param('models.svm', {})
        
        self.model = SVC(
            kernel=self.params.get('kernel', default_params.get('kernel', 'linear')),
            C=self.params.get('C', default_params.get('C', 1.0)),
            probability=self.params.get('probability', default_params.get('probability', True)),
            class_weight=self.params.get('class_weight', default_params.get('class_weight', None)),
            random_state=self.params.get('random_state', default_params.get('random_state', 42))
        )

    def plot_decision_boundary(self, X, y, title="SVM Decision Boundary"):
        """
        Visualizes the decision boundary, margins, and support vectors.
        
        NOTE: This method strictly requires 2D feature data.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")


        X = np.array(X)
        y = np.array(y)

        # Strict dimension check
        if X.shape[1] != 2:
            raise ValueError(
                f"Cannot visualize SVM boundary with {X.shape[1]} dimensions. "
                "Tip: Use PCA to reduce to 2D or select only 2 columns."
            )

        h = 0.02  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        plt.figure(figsize=(10, 6))

        # Calculate distance to hyperplane (decision_function)
        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['#FF9999', '#9999FF'])
        
        # Plot key lines:
        # Level -1 : Negative class margin (dashed)
        # Level  0 : Decision Boundary (solid)
        # Level  1 : Positive class margin (dashed)
        contours = plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
                               linestyles=['--', '-', '--'], 
                               colors='k', 
                               linewidths=[1, 2, 1])
        
        plt.clabel(contours, inline=True, fontsize=10, fmt='%1.0f')

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=60)
        
        sv = self.model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], s=200,
                    linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc="upper right")
        plt.grid(False)
        plt.show()