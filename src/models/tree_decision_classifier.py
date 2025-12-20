from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from src.models.base_classifier import BaseClassifier
from src.config import get_config


class DecisionTreeClassifier(BaseClassifier):
    """
    Specific implementation for Decision Tree.
    Inherits from BaseClassifier.
    """

    def _build_model(self):
        """
        Build the Decision Tree model with hyperparameters from config.yaml.

        Hyperparameters:
            - max_depth: Maximum depth of the tree (prevents overfitting)
            - criterion: 'gini' or 'entropy'
            - min_samples_split: Minimum samples required to split a node
            - min_samples_leaf: Minimum samples required at a leaf node
            - max_features: Number of features to consider for best split
            - class_weight: Weights associated with classes
            - random_state: Random state for reproducibility
        """
        config = get_config()

        # Get default hyperparameters from config if not provided in self.params
        default_params = config.get_param('models.decision_tree', {})

        self.model = SklearnDecisionTreeClassifier(
            max_depth=self.params.get(
                'max_depth', default_params.get('max_depth', 10)),
            criterion=self.params.get(
                'criterion', default_params.get('criterion', 'gini')),
            min_samples_split=self.params.get(
                'min_samples_split', default_params.get('min_samples_split', 2)),
            min_samples_leaf=self.params.get(
                'min_samples_leaf', default_params.get('min_samples_leaf', 1)),
            max_features=self.params.get(
                'max_features', default_params.get('max_features', None)),
            class_weight=self.params.get(
                'class_weight', default_params.get('class_weight', None)),
            random_state=self.params.get(
                'random_state', default_params.get('random_state', 42))
        )
