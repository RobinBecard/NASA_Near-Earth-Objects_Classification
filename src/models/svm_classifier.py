from sklearn.svm import SVC
from src.models.base_classifier import BaseModel
from src.config import get_config

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