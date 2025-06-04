import tensorflow as tf
import numpy as np
import os
from pathlib import Path

def prepare_mnist_data(data_path='data'):
    """
    Carrega o dataset MNIST, pré-processa e salva em arquivos .npy.
    """
    print("Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("Pré-processando dados...")
    x_train = x_train.astype('float32') / 255.0 # Normalizar os dados de treino [0, 1]
    x_test = x_test.astype('float32') / 255.0 # Normalizar os dados de teste [0, 1]

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Formato dos dados de treino: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Formato dos dados de teste: {x_test.shape}, Labels: {y_test.shape}")

    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Salvando dados pré-processados em {data_dir}...")
    np.save(data_dir / 'x_train.npy', x_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'x_test.npy', x_test)
    np.save(data_dir / 'y_test.npy', y_test)

    print("Preparação dos dados concluída.")
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    project_root = Path(__file__).parent
    data_save_path = project_root / 'data'
    prepare_mnist_data(data_path=data_save_path)