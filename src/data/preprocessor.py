import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_neo_data(df: pd.DataFrame) -> tuple:
    """Preprocess the NEO dataset for modeling.

    Args:
        df (pd.DataFrame): Raw NEO dataset.
    Returns:
        tuple: Preprocessed training and testing data (X_train, X_test, y_train, y_test).
    """
    
    num_features = ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 
                    'miss_distance', 'absolute_magnitude']

    X = df[num_features]
    y = df['hazardous']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test


