import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import our modules
from data_processing import load_heart_disease_data, preprocess_data
from model import create_cnn_model, train_model, evaluate_model, plot_training_history, plot_confusion_matrix

# Force CPU usage and disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set TensorFlow to use 32-bit floats
tf.keras.backend.set_floatx('float32')

def main():
    """
    Main function to run the heart disease detection pipeline.
    """
    print("Heart Disease Detection using CNN")
    print("=" * 40)
    
    # Step 1: Load the data
    print("\nStep 1: Loading data...")
    X, y = load_heart_disease_data()
    print(f"Data loaded: {X.shape[0]} samples with {X.shape[1]} features")
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing data...")
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    # Preprocess the data for CNN
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = preprocess_data(X_train_val, y_train_val)
    
    # Get validation set
    X_train_final, X_val_cnn, y_train_final, y_val_cnn = train_test_split(
        X_train_cnn, y_train_cnn, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train_final.shape}")
    print(f"Validation data shape: {X_val_cnn.shape}")
    print(f"Testing data shape: {X_test_cnn.shape}")
    
    # Step 3: Create the model
    print("\nStep 3: Creating CNN model...")
    input_shape = X_train_final.shape[1:]
    model = create_cnn_model(input_shape)
    model.summary()
    
    # Step 4: Train the model
    print("\nStep 4: Training the model...")
    history, trained_model = train_model(
        model, 
        X_train_final, 
        y_train_final, 
        X_val_cnn, 
        y_val_cnn,
        batch_size=16,  # Smaller batch size
        epochs=20       # Fewer epochs
    )
    
    # Step 5: Evaluate the model
    print("\nStep 5: Evaluating the model...")
    results, confusion_matrix, y_pred_prob = evaluate_model(trained_model, X_test_cnn, y_test_cnn)
    
    # Step 6: Plot results
    print("\nStep 6: Plotting results...")
    plot_training_history(history)
    plot_confusion_matrix(confusion_matrix)
    
    # Step 7: Save the model
    print("\nStep 7: Saving the model...")
    trained_model.save('best_heart_disease_model.h5')
    
    print("\nHeart Disease Detection Model Training Complete!")
    print(f"Model saved as 'best_heart_disease_model.h5'")

if __name__ == "__main__":
    # Run the main function
    main() 