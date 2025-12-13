from sklearn.neural_network import MLPClassifier
from src.models.base_classifier import BaseModel
from src.config import get_config

class NNClassifier(BaseModel):
    """
    Specific implementation for Multilayer Perceptron (Neural Network).
    Inherits from BaseModel.
    """

    def _build_model(self):
        """
        Builds the MLPClassifier model with hyperparameters from config.yaml.

        Key hyperparameters:
            - hidden_layer_sizes: Tuple defining the architecture (e.g., (100, 50) for 2 hidden layers).
            - activation: Activation function ('relu', 'tanh', 'logistic').
            - solver: The optimizer ('adam', 'sgd').
            - alpha: L2 regularization parameter.
            - max_iter: Maximum number of epochs.
        """
        config = get_config()
        
        # Retrieve default parameters from the yaml file
        # If the section does not exist, use an empty dictionary {}
        default_params = config.get_param('models.neural_network', {})
        
        self.model = MLPClassifier(
            hidden_layer_sizes=self.params.get('hidden_layer_sizes', default_params.get('hidden_layer_sizes', (100,))),
            activation=self.params.get('activation', default_params.get('activation', 'relu')),
            solver=self.params.get('solver', default_params.get('solver', 'adam')),
            alpha=self.params.get('alpha', default_params.get('alpha', 0.0001)),
            learning_rate_init=self.params.get('learning_rate_init', default_params.get('learning_rate_init', 0.001)),
            max_iter=self.params.get('max_iter', default_params.get('max_iter', 200)),
            early_stopping=self.params.get('early_stopping', default_params.get('early_stopping', True)),
            random_state=self.params.get('random_state', default_params.get('random_state', 42))
        )