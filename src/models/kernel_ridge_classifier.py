from sklearn.kernel_ridge import KernelRidge
from src.models.base_classifier import BaseModel
from src.config import get_config
import numpy as np
import matplotlib.pyplot as plt

class KernelRidgeClassifier(BaseModel):
    """
    Specific implementation for Kernel Ridge Regression used as a Classifier.
    Inherits from BaseModel.
    
    Note: KRR is technically a regression algorithm. 
    We adapt it for classification by thresholding the output.
    """

    def _build_model(self):
        """
        Build the KernelRidge model with hyperparameters from config.yaml.
        
        Hyperparameters:
            - alpha: Regularization strength (Small alpha = complex model, Large alpha = simple model).
            - kernel: Kernel mapping ('rbf', 'linear', 'polynomial', 'sigmoid').
            - gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
            - degree: Degree of the polynomial kernel.
        """
        config = get_config()
        default_params = config.get_param('models.kernel_ridge', {})
        
        self.model = KernelRidge(
            alpha=self.params.get('alpha', default_params.get('alpha', 1.0)),
            kernel=self.params.get('kernel', default_params.get('kernel', 'rbf')),
            gamma=self.params.get('gamma', default_params.get('gamma', None)),
            degree=self.params.get('degree', default_params.get('degree', 3))
        )

    def predict(self, X):
        """
        Overridden method.
        Since KernelRidge returns continuous values (regression), 
        we must threshold them to get binary classes (0 or 1).
        """
        # Get raw regression scores
        scores = self.model.predict(X)
        
        # Threshold at 0.5 (assuming labels are 0 and 1)
        # If the regression predicts > 0.5, it's class 1, else class 0
        return np.where(scores >= 0.5, 1, 0)

    def plot_prediction_surface(self, X, y, title="KRR Prediction Surface"):
        """
        Visualizes the continuous regression surface learned by the kernel.
        Unlike a simple boundary, this shows the "confidence" (height) of the model everywhere.
        
        NOTE: Requires 2D features.
        """
        if self.model is None:
            print("Error: Model not trained.")
            return

        X = np.array(X)
        y = np.array(y)

        if X.shape[1] != 2:
            print(f"⚠️ Visualization requires 2D features. Got {X.shape[1]}D.")
            return

        # 1. Create meshgrid
        h = 0.05 # Step size (larger than others to calculate faster)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 2. Predict raw values (Z is continuous here, not just 0/1)
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 7))

        # 3. Plot the continuous surface as a heatmap
        # Blue areas are predicted < 0.5 (Class 0), Red areas > 0.5 (Class 1)
        contour = plt.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.8)
        plt.colorbar(contour, label="Regression Output (Score)")

        # 4. Add the specific decision line (where score = 0.5)
        plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=2)

        # 5. Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors='white', s=60)
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()