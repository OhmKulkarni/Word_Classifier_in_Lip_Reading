# --- Import Libraries ---
import os
from gc import callbacks

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Bidirectional, \
    LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import visualkeras
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt

# --- Mixed Precision Training ---
set_global_policy('mixed_float16')

# Ensure GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Clear any existing GPU memory
from tensorflow.keras import backend as K
K.clear_session()


# --- Lazy Loading of Data ---
def lazy_load_data(folder):
    """Generator to load data sample-by-sample for memory efficiency."""
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            label = filename.split('_')[2]  # Extract label from filename
            data = np.load(file_path)
            yield data, label


# Prepare dataset function
def prepare_dataset(folder, label_encoder=None):
    """Load data lazily and create dataset."""
    print(f"Preparing dataset from: {folder}...")
    X, y = zip(*lazy_load_data(folder))
    X, y = np.array(X), np.array(y)

    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = label_encoder.transform(y)

    print(f"Data loaded: {len(X)} samples, {X[0].shape[0]} frames per video.")
    return X[..., np.newaxis], y, label_encoder  # Add channel for grayscale


# Create TF Dataset API for efficient data loading
def create_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# --- CNN + BiLSTM Model Definition with L2 Regularization ---
def create_model_with_bilstm(input_shape, num_classes):
    model = Sequential()

    # L2 Regularization strength
    l2_strength = 1e-4  # You can adjust this value

    # 1st Convolutional Block
    model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same',
                     input_shape=input_shape, kernel_regularizer=l2(l2_strength)))  # Reduced filters from 32 to 16
    model.add(MaxPooling3D((2, 2, 1)))  # Adjusted pooling size to (2, 2, 1)
    model.add(BatchNormalization())

    # 2nd Convolutional Block
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(l2_strength)))  # Reduced filters from 64 to 32
    model.add(MaxPooling3D((2, 2, 1)))  # Adjusted pooling size to (2, 2, 1)
    model.add(BatchNormalization())

    # Flatten and Reshape for LSTM input
    model.add(Flatten())
    model.add(tf.keras.layers.Reshape((input_shape[0], -1)))

    # Bidirectional LSTM Layer
    model.add(Bidirectional(LSTM(64, return_sequences=False,
                                 kernel_regularizer=l2(l2_strength))))  # Reduced LSTM units from 128 to 64

    # Dense Layers for Classification
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_strength)))  # Reduced units from 256 to 128
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_strength)))  # Reduced units from 64 to 32
    model.add(Dense(num_classes, activation='softmax', dtype='float32'))  # Output remains float32 for stability

    # Compile the model
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    plot_model(
        model,
        to_file='Documentation_Images/CNN+BiLSTM/CNN+BiLSTM_Model.png',
        show_shapes=True,
        show_layer_names=True
    )

    visualkeras.layered_view(model, to_file='Documentation_Images/CNN+BiLSTM/CNN+BiLSTM_3DModel.png',legend=True)

    return model


# --- Print Label Distribution ---
def print_label_distribution(y, dataset_name):
    counts = Counter(y)
    print(f"Label distribution in {dataset_name}:")
    for label, count in counts.items():
        print(f"Label {label}: {count} samples")


# --- Plot Training History ---
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.savefig('Documentation_Images/CNN+BiLSTM/CNN+BiLSTM_TrainingHistory.png')
    plt.close()

def plot_loss_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Save and close the figure to prevent memory issues
    plt.savefig('Documentation_Images/CNN+BiLSTM/CNN+BiLSTM_LossHistory.png')
    plt.close()



# --- Test Accuracy Callback ---
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, n=2):
        super().__init__()
        self.test_dataset = test_dataset
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            test_loss, test_acc = self.model.evaluate(self.test_dataset, verbose=0)
            print(f"\nTest Accuracy at epoch {epoch + 1}: {test_acc:.4f}")


# --- Gradient Norm Logging Callback ---
class GradientNormLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, train_dataset):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset

    def on_epoch_end(self, epoch, logs=None):
        # Create GradientTape to track computations
        for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
            with tf.GradientTape() as tape:
                logits = self.model(x_batch_train, training=True)  # Forward pass
                loss_value = self.model.compiled_loss(
                    y_batch_train, logits, regularization_losses=self.model.losses
                )

            # Get the gradients of the trainable variables with respect to the loss
            gradients = tape.gradient(loss_value, self.model.trainable_weights)

            # Log the gradient norms for each layer
            for layer, gradient in zip(self.model.layers, gradients):
                if gradient is not None:
                    gradient_norm = tf.norm(gradient).numpy()
                    print(f"Gradient norm for layer {layer.name} at epoch {epoch + 1}: {gradient_norm:.6f}")
            break  # Only log for the first batch to avoid excessive computation


# --- Main Training Function ---
def train_model(preprocessed_folder, test_folder, test_size=0.2, batch_size=8, epochs=30, test_eval_interval=1):
    # Load and prepare datasets
    X, y, label_encoder = prepare_dataset(preprocessed_folder)
    print_label_distribution(y, "Complete Dataset")

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Create tf.data datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size)

    # Load test dataset
    X_test, y_test, _ = prepare_dataset(test_folder, label_encoder)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size)

    # Print label distributions for validation and test sets
    print_label_distribution(y_val, "Validation Dataset")
    # print_label_distribution(y_test, "Test Dataset")

    # Define input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    num_classes = len(np.unique(y))

    # Create the model
    model = create_model_with_bilstm(input_shape, num_classes)
    print(model.summary())

    # --- Callback: Save the Best Model ---
    checkpoint_callback = ModelCheckpoint(
        filepath='models/CNN+BiLSTM_Model.h5',  # File to save the best model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Save when 'val_accuracy' is maximized
        save_best_only=True,  # Save only the best model
        verbose=1  # Print a message when the model is saved
    )

    # Define callbacks
    test_accuracy_callback = TestAccuracyCallback(test_dataset, n=test_eval_interval)
    gradient_logging_callback = GradientNormLoggingCallback(model, train_dataset)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_callback]
        # callbacks=[test_accuracy_callback, gradient_logging_callback]
    )

    # Evaluate the model on the test dataset
    # test_loss, test_acc = model.evaluate(test_dataset)
    # print(f"Final Test Loss: {test_loss}, Final Test Accuracy: {test_acc}")

    # Plot training history
    plot_training_history(history)
    plot_loss_history(history)

    K.clear_session()
    return model, history


# --- Run Training ---
preprocessed_folder = 'preprocessed_data'
test_folder = 'preprocessed_test_data'

model, history = train_model(preprocessed_folder, test_folder, epochs=50, test_eval_interval=1)
