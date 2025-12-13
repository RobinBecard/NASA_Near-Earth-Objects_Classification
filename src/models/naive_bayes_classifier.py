from sklearn.naive_bayes import GaussianNB
from src.models.base_classifier import BaseModel
from src.config import get_config
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class NaiveBayesClassifier(BaseModel):
    """
    Specific implementation for Gaussian Naive Bayes.
    Inherits from BaseModel.
    """

    def _build_model(self):
        """
        Build the GaussianNB model with hyperparameters from config.yaml.
        
        Hyperparameters:
            - var_smoothing: Portion of the largest variance of all features 
                             that is added to variances for calculation stability.
        """
        config = get_config()
        default_params = config.get_param('models.naive_bayes', {})
        
        self.model = GaussianNB(
            var_smoothing=self.params.get('var_smoothing', default_params.get('var_smoothing', 1e-9))
        )

    def plot_distributions(self, feature_indices=None, feature_names=None):
        """
        Visualizes the Gaussian distributions learned by the model for each class.
        This shows how the model separates classes based on feature density.

        Args:
            feature_indices (list): Indices of features to plot (e.g., [0, 1]). 
                                    If None, plots first 4 features.
            feature_names (list): Names of the features for the legend.
        """
        if self.model is None:
            print("Error: Model has not been trained. Call train() before plotting distributions.")
            return

        # Check if the model has learned attributes (theta_ = means, var_ = variances)
        if not hasattr(self.model, 'theta_'):
            print("Error: Model must be trained to visualize distributions.")
            return

        n_features = self.model.theta_.shape[1]
        if feature_indices is None:
            feature_indices = range(min(n_features, 4)) # Default to first 4
        
        n_plots = len(feature_indices)
        classes = self.model.classes_
        
        cols = 2
        rows = (n_plots + 1) // cols
        plt.figure(figsize=(12, 4 * rows))

        for i, feature_idx in enumerate(feature_indices):
            plt.subplot(rows, cols, i + 1)
            
            for class_idx, class_label in enumerate(classes):
                mean = self.model.theta_[class_idx, feature_idx]
                var = self.model.var_[class_idx, feature_idx]
                std = np.sqrt(var)
                
                x = np.linspace(mean - 4*std, mean + 4*std, 100)
                y = stats.norm.pdf(x, mean, std)
                
                label = f"Class {class_label}"
                plt.plot(x, y, label=label, linewidth=2)
                plt.fill_between(x, y, alpha=0.2)

            f_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
            plt.title(f"Distribution: {f_name}")
            plt.xlabel("Feature Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()