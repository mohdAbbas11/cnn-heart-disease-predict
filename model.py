import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Monkey patch the signbit function to avoid the overflow error
def safe_signbit(x):
    # A safer implementation that avoids using large integer constants
    return tf.cast(tf.less(x, 0), tf.bool)

# Apply the monkey patch if using TensorFlow 2.16+
if tf.__version__.startswith('2.16') or tf.__version__.startswith('2.17') or tf.__version__.startswith('2.18') or tf.__version__.startswith('2.19'):
    try:
        import keras.src.backend.tensorflow.numpy as knp
        original_signbit = knp.signbit
        knp.signbit = safe_signbit
        print("Applied signbit patch for TensorFlow", tf.__version__)
    except:
        print("Could not apply signbit patch, but continuing anyway")

# Use tf.keras directly
def create_cnn_model(input_shape, num_classes=2):
    """
    Create a simplified CNN model for heart disease detection.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.models.Sequential()
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model with basic settings
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train the CNN model with early stopping.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        
    Returns:
        Training history and trained model
    """
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data and labels
        
    Returns:
        Evaluation metrics
    """
    # Evaluate the model
    results = model.evaluate(X_test, y_test, verbose=1)
    
    # Print results
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])
    
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate additional metrics manually
    auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print("Test AUC:", auc)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Disease', 'Disease']))
    
    return results, cm, y_pred_prob

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history from model.fit()
    """
    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(cm):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
    """
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show() 