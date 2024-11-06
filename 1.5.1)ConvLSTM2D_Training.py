import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Reshape, Layer, Softmax
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import visualkeras

# --- Mixed Precision Training ---
set_global_policy('mixed_float16')

# Ensure GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import backend as K
K.clear_session()

# --- Temporal Attention Layer ---
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
            dtype='float16'  # Added explicit dtype
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float16')  # Added explicit casting
        scores = tf.matmul(inputs, self.attention_weights)
        scores = Softmax(dtype='float16')(scores)  # Added explicit dtype
        weighted_inputs = tf.multiply(inputs, scores)  # Changed from * to tf.multiply
        return tf.reduce_sum(weighted_inputs, axis=1)

    def compute_output_shape(self, input_shape):  # Added this method
        return (input_shape[0], input_shape[2])

# --- Lazy Loading of Data ---
def lazy_load_data(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            label = filename.split('_')[2]
            data = np.load(file_path)
            yield data, label

def prepare_dataset(folder, label_encoder=None):
    X, y = zip(*lazy_load_data(folder))
    X, y = np.array(X), np.array(y)

    # Encode labels using the provided encoder or create a new one
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = label_encoder.transform(y)

    return X[..., np.newaxis], y, label_encoder

# --- TF Dataset API ---
def create_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- 3D CNN Model with Temporal Attention ---
def create_model_with_attention(input_shape, num_classes):
    model = Sequential()

    # 1st Convolutional Block
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(BatchNormalization())

    # 2nd Convolutional Block
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(BatchNormalization())

    # 3rd Convolutional Block
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(BatchNormalization())

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Reshape((-1, 256)))
    model.add(TemporalAttention())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax', dtype='float32'))

    # Optimizer with gradient accumulation
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    plot_model(
        model,
        to_file='Documentation_Images/ConvLSTM2D/ConvLSTM2D_Model.png',
        show_shapes=True,
        show_layer_names=True
    )

    visualkeras.layered_view(model, to_file='Documentation_Images/ConvLSTM2D/ConvLSTM2D_3DModel.png', legend=True)

    return model

# --- Helper Function: Print Label Distribution ---
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
    # Save and close the figure to prevent memory issues
    plt.savefig('Documentation_Images/ConvLSTM2D/ConvLSTM2D_TrainingHistory.png')
    plt.close()

def plot_loss_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Save and close the figure to prevent memory issues
    plt.savefig('Documentation_Images/ConvLSTM2D/ConvLSTM2D_LossHistory.png')
    plt.close()

# --- Custom Callback to Evaluate Test Accuracy Every n Epochs ---
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, n=2):
        super().__init__()
        self.test_dataset = test_dataset
        self.n = n  # Evaluate every 'n' epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:  # Run every 'n' epochs
            test_loss, test_acc = self.model.evaluate(self.test_dataset, verbose=0)
            print(f"\nTest Accuracy at epoch {epoch + 1}: {test_acc:.4f}")

# --- Main Training Function ---
def train_model(preprocessed_folder, test_folder, test_size=0.2, batch_size=4, epochs=100, test_eval_interval=2):
    # Load the data for training and validation
    X, y, label_encoder = prepare_dataset(preprocessed_folder)

    # Print label distribution for the full dataset
    print_label_distribution(y, "Complete Dataset")

    # Split the data (stratified split) for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Create tf.data datasets for training and validation
    train_dataset = create_tf_dataset(X_train, y_train, batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size)

    # Load the final test data (from a different folder) using the same label encoder
    X_test, y_test, _ = prepare_dataset(test_folder, label_encoder)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size)

    # Print label distribution for validation and test sets
    # print_label_distribution(y_val, "Validation Dataset")
    # print_label_distribution(y_test, "Test Dataset")

    # Define input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    num_classes = len(np.unique(y))

    # Create the model
    model = create_model_with_attention(input_shape, num_classes)
    print(model.summary())

    # --- Callback: Save the Best Model ---
    checkpoint_callback = ModelCheckpoint(
        filepath='models/ConvLSTM2D_Model.h5',  # File to save the best model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Save when 'val_accuracy' is maximized
        save_best_only=True,  # Save only the best model
        verbose=1  # Print a message when the model is saved
    )

    # Create the custom callback to evaluate test accuracy every `test_eval_interval` epochs
    # test_accuracy_callback = TestAccuracyCallback(test_dataset, n=test_eval_interval)

    # Train the model using the built-in fit function with validation data and the custom callback
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=len(X_train) // batch_size,
        validation_steps=len(X_val) // batch_size,
        callbacks = [checkpoint_callback]
        # callbacks=[test_accuracy_callback]
    )

    # Final evaluation on the test dataset after all epochs
    # test_loss, test_acc = model.evaluate(test_dataset)
    # print(f"Final Test loss: {test_loss}, Final Test accuracy: {test_acc}")

    # Plot training history
    plot_training_history(history)
    plot_loss_history(history)

    K.clear_session()
    return model, history

# --- Run the Training Process ---
preprocessed_folder = 'preprocessed_data'
test_folder = 'preprocessed_test_data'
model, history = train_model(preprocessed_folder, test_folder, epochs=30, test_eval_interval=1)  # Specify epochs here
