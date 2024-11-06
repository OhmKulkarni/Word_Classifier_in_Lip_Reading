import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from collections import Counter
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import visualkeras
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import pandas as pd

# --- Mixed Precision Training ---
set_global_policy('mixed_float16')

# Optimize GPU memory usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Set memory limit to prevent OOM
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 4)]  # 4GB limit
    )

# Clear any existing GPU memory
tf.keras.backend.clear_session()

AUTOTUNE = tf.data.experimental.AUTOTUNE


class OptimizedDataGenerator:
    """Memory-efficient data generator using tf.data.Dataset"""

    def __init__(self, folder, batch_size=32, shuffle=True):
        self.folder = folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = [f for f in os.listdir(folder) if f.endswith(".npy")]
        self.label_encoder = LabelEncoder()

        # Extract labels from filenames
        labels = [f.split('_')[2] for f in self.filenames]
        self.encoded_labels = self.label_encoder.fit_transform(labels)

        # Create tf.data.Dataset
        self.dataset = self._create_dataset()

    def _parse_file(self, filename, label):
        # Load and process file
        data = tf.numpy_function(
            lambda x: np.load(os.path.join(self.folder, x.decode())).astype(np.float16),
            [filename],
            tf.float16
        )
        # Add channel dimension
        data = tf.expand_dims(data, axis=-1)
        return data, label

    def _create_dataset(self):
        # Create dataset from filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.filenames, self.encoded_labels)
        )

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.filenames))

        # Optimize pipeline
        dataset = dataset.map(
            self._parse_file,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.cache()  # Cache data in memory
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)  # Prefetch next batch

        return dataset

    def get_label_encoder(self):
        return self.label_encoder

    @property
    def total_samples(self):
        return len(self.filenames)


def create_model(input_shape, num_classes):
    model = Sequential()

    # 1st Convolutional Block - Using strides instead of MaxPooling
    model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(BatchNormalization())

    # 2nd Convolutional Block
    model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())

    # 3rd Convolutional Block
    model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', dtype='float32'))

    # Use a larger learning rate with decay
    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    optimizer = Adam(learning_rate=lr_schedule)

    # Enable mixed precision
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    try:
        plot_model(model, to_file='Documentation_Images/3DCNN/3DCNN_Model.png',
                   show_shapes=True, show_layer_names=True)
        visualkeras.layered_view(model, to_file='Documentation_Images/3DCNN/3DCNN_3DModel.png',
                                 legend=True)
    except Exception as e:
        print(f"Warning: Could not generate model visualization: {e}")

    return model


# Helper functions remain unchanged
def print_label_distribution(y, dataset_name):
    counts = Counter(y)
    print(f"Label distribution in {dataset_name}:")
    for label, count in counts.items():
        print(f"Label {label}: {count} samples")


def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('Documentation_Images/3DCNN/3DCNN_TrainingHistory.png')
    plt.close()


def plot_loss_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('Documentation_Images/3DCNN/3DCNN_LossHistory.png')
    plt.close()


# Optimized callbacks
class OptimizedTestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, n=2):
        super().__init__()
        self.test_dataset = test_dataset
        self.n = n
        self._test_data = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            test_loss, test_acc = self.model.evaluate(self.test_dataset.dataset,
                                                      verbose=0)
            print(f"\nTest Accuracy at epoch {epoch + 1}: {test_acc:.4f}")


class OptimizedDetailedMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, label_encoder, evaluation_interval=5):
        super().__init__()
        self.test_dataset = test_dataset
        self.label_encoder = label_encoder
        self.evaluation_interval = evaluation_interval
        self.class_names = label_encoder.classes_

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.evaluation_interval == 0:
            y_pred = []
            y_true = []

            for x_batch, y_batch in self.test_dataset.dataset:
                pred = self.model.predict(x_batch, verbose=0)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(y_batch.numpy())

            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )

            metrics_df = pd.DataFrame({
                'Class': self.class_names,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })

            print(f"\n=== Detailed Metrics at Epoch {epoch + 1} ===")
            print(metrics_df.to_string(index=False))

            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            os.makedirs('Documentation_Images/3DCNN/confusion_matrices', exist_ok=True)
            plt.savefig(f'Documentation_Images/3DCNN/confusion_matrices/confusion_matrix_epoch_{epoch + 1}.png')
            plt.close()

            os.makedirs('Documentation_Images/3DCNN/metrics', exist_ok=True)
            metrics_df.to_csv(f'Documentation_Images/3DCNN/metrics/metrics_epoch_{epoch + 1}.csv',
                              index=False)


def train_model(preprocessed_folder, test_folder, test_size=0.2, batch_size=64,
                epochs=3, test_eval_interval=2, evaluation_interval=1):
    # Create directories for train and validation data
    train_dir = os.path.join(preprocessed_folder, 'train')
    val_dir = os.path.join(preprocessed_folder, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all files and their labels
    all_files = [f for f in os.listdir(preprocessed_folder) if f.endswith(".npy")]
    labels = [f.split('_')[2] for f in all_files]

    # Initialize label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split files for training and validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, encoded_labels, test_size=test_size, stratify=encoded_labels,
        random_state=42
    )

    # Create symbolic links or copy files
    for files, directory in [(train_files, train_dir), (val_files, val_dir)]:
        for file in files:
            src = os.path.join(preprocessed_folder, file)
            dst = os.path.join(directory, file)
            if not os.path.exists(dst):
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(src, dst)
                else:  # Unix-like
                    os.symlink(src, dst)

    # Create optimized data generators
    train_generator = OptimizedDataGenerator(train_dir, batch_size=batch_size)
    val_generator = OptimizedDataGenerator(val_dir, batch_size=batch_size)
    test_generator = OptimizedDataGenerator(test_folder, batch_size=batch_size)

    # Print distributions
    print_label_distribution(train_labels, "Training Dataset")
    print_label_distribution(val_labels, "Validation Dataset")
    print_label_distribution(test_generator.encoded_labels, "Test Dataset")

    # Get input shape from a sample file
    sample_file = os.path.join(preprocessed_folder, all_files[0])
    input_shape = (*np.load(sample_file).shape, 1)

    # Create and compile model
    num_classes = len(label_encoder.classes_)
    model = create_model(input_shape, num_classes)
    print(model.summary())

    # Create callbacks
    detailed_metrics_callback = OptimizedDetailedMetricsCallback(
        test_dataset=test_generator,
        label_encoder=label_encoder,
        evaluation_interval=evaluation_interval
    )

    test_accuracy_callback = OptimizedTestAccuracyCallback(
        test_generator,
        n=test_eval_interval
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='models/3DCNN_Model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    print("Starting training...")
    history = model.fit(
        train_generator.dataset,
        validation_data=val_generator.dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_callback, test_accuracy_callback,
                   detailed_metrics_callback],
        workers=4,
        use_multiprocessing=True
    )

    # Final evaluation
    test_loss, test_acc = model.evaluate(test_generator.dataset)
    print(f"Final Test loss: {test_loss}, Final Test accuracy: {test_acc}")

    # Plot training history
    plot_training_history(history)
    plot_loss_history(history)

    tf.keras.backend.clear_session()
    return model, history


# Run the training process
if __name__ == "__main__":
    preprocessed_folder = 'preprocessed_data'
    test_folder = 'preprocessed_test_data'
    model, history = train_model(
        preprocessed_folder,
        test_folder,
        epochs=50,
        test_eval_interval=1,
        evaluation_interval=1,
        batch_size=64  # Increased batch size
    )