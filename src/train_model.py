import tensorflow as tf
import numpy as np
from pathlib import Path

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
    except RuntimeError as e:
        print(f"⚠️ Error during GPU configuration: {e}")
else:
    print("⚠️ No GPU found. TensorFlow will run on CPU. For GPU acceleration, ensure TensorFlow-GPU is installed and NVIDIA drivers/CUDA are correctly set up.")

# 28x28 only grayscale images, 10 classes (0-9)
def get_model_cnn_simple(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name="cnn_simple")
    return model

def get_model_cnn_deeper(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name="cnn_deeper")
    return model

def get_model_cnn_dropout(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name="cnn_dropout")
    return model

def train_and_save_model(model_fn, model_name, x_train, y_train, x_test, y_test, epochs=10, models_save_dir='keras_models'):
    print(f"\n--- Training model: {model_name} ---")
    model = model_fn()
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)

    print(f"Evaluating Keras model: {model_name}")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Keras Accuracy ({model_name}): {accuracy:.4f}")

    save_dir = Path(models_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}.h5"
    try:
        model.save(model_path)
        print(f"Model {model_name} saved to {model_path}")
    except Exception as e:
        print(f"Error saving model {model_name} in H5 format: {e}. Saving in TensorFlow SavedModel format instead.")
        model_path_tf = save_dir / model_name
        model.save(model_path_tf) 
        print(f"Model {model_name} saved to {model_path_tf}")
        return str(model_path_tf), accuracy 
        
    return str(model_path), accuracy

if __name__ == '__main__':
    script_dir = Path(__file__)
    project_root = script_dir.parent
    
    data_load_path = project_root / 'data'
    models_save_path = project_root / 'models' / 'keras_models'

    print(f"Project root (assumed): {project_root}")
    print(f"Loading data from: {data_load_path}")
    print(f"Saving models to: {models_save_path}")

    models_save_path.mkdir(parents=True, exist_ok=True)

    print("Loading pre-processed data...")
    try:
        x_train = np.load(data_load_path / 'x_train.npy')
        y_train = np.load(data_load_path / 'y_train.npy')
        x_test = np.load(data_load_path / 'x_test.npy')
        y_test = np.load(data_load_path / 'y_test.npy')
    except FileNotFoundError as e:
        print("Please ensure 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy' are present in the 'data' directory.")
        exit()

    model_definitions = [
        (get_model_cnn_simple, "cnn_simple", 10),
        (get_model_cnn_deeper, "cnn_deeper", 15),
        (get_model_cnn_dropout, "cnn_dropout", 20)
    ]

    trained_models_info = []
    for model_fn, name, epochs_count in model_definitions:
        path, acc = train_and_save_model(model_fn, name, x_train, y_train, x_test, y_test,
                                         epochs=epochs_count, models_save_dir=models_save_path)
        trained_models_info.append({'name': name, 'path': path, 'accuracy': acc})

    print("\n--- Keras Training Summary ---")
    for info in trained_models_info:
        print(f"Model: {info['name']}, Accuracy: {info['accuracy']:.4f}, Saved to: {info['path']}")