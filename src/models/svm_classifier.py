from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    A Support Vector Machine (SVM) classifier wrapper.

    Inheritance Rationale:
    ----------------------
    1. BaseEstimator:
       - Inherited Methods: get_params(), set_params()
       - Utility: Enables seamless integration with Scikit-Learn tools such as 
         GridSearchCV and Pipelines. It allows these tools to automatically 
         inspect and modify the model's hyperparameters (kernel, C, etc.).

    2. ClassifierMixin:
       - Inherited Methods: score(X, y)
       - Utility: Provides the standard implementation of the score() method, 
         which calculates the mean accuracy on the given test data and labels. 
         This ensures the custom class behaves consistently with other standard 
         classifiers.


    Parameters:    
    -----------
    kernel : str, optional (default='linear')
        Specifies the kernel type to be used in the algorithm. It can be 'linear', 'poly', 'rbf', 'sigmoid', etc.
    C : float, optional (default=1.0)
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    probability : bool, optional (default=True)
        Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
    class_weight : dict or 'balanced', optional (default=None)
        Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
    
    Methods:    
    --------
    fit(X, y):
        Fit the SVM model according to the given training data.
    predict(X):
        Perform classification on samples in X.
    predict_proba(X):
        Return probability estimates for the test data X. Required for ROC Curve and AUC calculation.
    
    """
    
    def __init__(self, kernel='linear', C=1.0, probability=True, class_weight=None):
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self.class_weight = class_weight # Handles class imbalance (e.g., 'balanced')
        self.model = None

    def fit(self, X, y):
        # Initialize and train the internal SVC model with current parameters
        self.model = SVC(
            kernel=self.kernel, 
            C=self.C, 
            probability=self.probability, 
            class_weight=self.class_weight
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise Exception("Model not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        # Required for ROC Curve and AUC calculation
        if self.model is None:
            raise Exception("Model not trained yet.")
        return self.model.predict_proba(X)