import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def load_heart_disease_data():
    """
    Load the UCI Heart Disease dataset.
    Returns preprocessed X and y data.
    """
    # URL for the UCI Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load the dataset
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print("Dataset loaded successfully from URL")
    except Exception as e:
        print(f"Error loading from URL: {e}")
        print("Attempting to load from local file or alternative source...")
        
        # Alternative: Use the dataset from scikit-learn
        from sklearn.datasets import fetch_openml
        heart = fetch_openml(name='heart', version=1)
        df = pd.DataFrame(heart.data, columns=heart.feature_names)
        df['target'] = heart.target
        print("Dataset loaded from scikit-learn")
    
    # Clean the data
    df = df.replace('?', np.nan).dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess the data for the CNN model.
    
    Args:
        X: Features dataframe
        y: Target series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train_reshaped, X_test_reshaped, y_train, y_test
    """
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Convert y to int
    y = y.astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure data is float32
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    
    # Reshape for CNN (samples, height, width, channels)
    # For tabular data, we'll create a 2D representation
    n_features = X_train_scaled.shape[1]
    
    # Calculate dimensions for a roughly square 2D representation
    height = int(np.ceil(np.sqrt(n_features)))
    width = int(np.ceil(n_features / height))
    
    # Pad the data to fit the square shape
    X_train_padded = np.zeros((X_train_scaled.shape[0], height * width), dtype=np.float32)
    X_test_padded = np.zeros((X_test_scaled.shape[0], height * width), dtype=np.float32)
    
    X_train_padded[:, :n_features] = X_train_scaled
    X_test_padded[:, :n_features] = X_test_scaled
    
    # Reshape to 2D + channel for CNN
    X_train_reshaped = X_train_padded.reshape(-1, height, width, 1)
    X_test_reshaped = X_test_padded.reshape(-1, height, width, 1)
    
    # Convert targets to categorical format
    y_train = tf.keras.utils.to_categorical(y_train.astype(np.int32), num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes=2)
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test

def load_and_preprocess_single_sample(data_dict):
    """
    Load and preprocess a single sample for prediction.
    
    Args:
        data_dict: Dictionary containing feature values
        
    Returns:
        Preprocessed data ready for CNN model
    """
    # Expected features
    expected_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Create a DataFrame with the input data
    sample = pd.DataFrame([data_dict])
    
    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in sample.columns:
            raise ValueError(f"Missing feature: {feature}")
    
    # Convert to numeric
    sample = sample.apply(pd.to_numeric, errors='coerce')
    
    # Load the scaler (in a real application, this would be saved during training)
    # For this example, we'll use a simple standardization
    scaler = StandardScaler()
    
    # Fit the scaler on the sample (in a real application, use the saved scaler)
    sample_scaled = scaler.fit_transform(sample)
    
    # Ensure data is float32
    sample_scaled = sample_scaled.astype(np.float32)
    
    # Reshape for CNN (samples, height, width, channels)
    n_features = sample_scaled.shape[1]
    
    # Calculate dimensions for a roughly square 2D representation
    height = int(np.ceil(np.sqrt(n_features)))
    width = int(np.ceil(n_features / height))
    
    # Pad the data to fit the square shape
    sample_padded = np.zeros((1, height * width), dtype=np.float32)
    sample_padded[:, :n_features] = sample_scaled
    
    # Reshape to 2D + channel for CNN
    sample_reshaped = sample_padded.reshape(-1, height, width, 1)
    
    return sample_reshaped

if __name__ == "__main__":
    # Test the data loading and preprocessing
    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}") 