import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def read_dataset(csv_train_path, csv_test_path):
    """
    Reads two CSV files with the MNIST data for training and testing
    INPUT:
        csv_train_path: Path of the CSV file with training data (str)
        csv_test_path: Path of the CSV file with training data (str)
    OUTPUT:
        df_train: Training dataframe (pandas.core.frame.DataFrame)
        df_test: Test dataframe (pandas.core.frame.DataFrame)
    """

    df_train = pd.read_csv(csv_train_path, sep=',', header=0)
    df_test = pd.read_csv(csv_test_path, sep=',', header=0)

    return df_train, df_test


def plot_overview_dataset(df):
    """
    Shows a sample of all classes of handwritten digits
    INPUT:
        df: Original dataframe (pandas.core.frame.DataFrame)
    """
    
    data_plot = df.to_numpy()
    X_plot = data_plot[:, 1:].reshape(-1, 28, 28)
    y_plot = data_plot[:, 0].astype(int)
    
    n_per_digit = 10
    
    cols = int(np.ceil(np.sqrt(n_per_digit)))
    rows = int(np.ceil(n_per_digit / cols))
    
    fig, axes = plt.subplots(
        10, rows * cols,
        figsize=(rows * cols, 10),
        dpi=100
    )
    
    for digit in range(10):
        idx = np.where(y_plot == digit)[0][:n_per_digit]
        for j in range(rows * cols):
            ax = axes[digit, j]
            ax.axis('off')
            if j < len(idx):
                ax.imshow(X_plot[idx[j]], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            else:
                ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def split_dataset(data, ratio):
    """
    Randomly splits original training dataframe into a training set and a validation set
    INPUT:
        data: Original training dataframe (pandas.core.frame.DataFrame)
        ratio: Proportion of instances from the training dataframe that go into the validation set (float)
    OUTPUT:
        df_train_new: New training dataframe (pandas.core.frame.DataFrame)
        df_validation_new: New validation dataframe (pandas.core.frame.DataFrame)
    """
    
    shuffled_indices = np.random.permutation(len(data))
    sub_set_size = int(len(data)*ratio)
    train_indices = shuffled_indices[sub_set_size:]
    validation_indices = shuffled_indices[:sub_set_size]
    df_train_new = data.iloc[train_indices]
    df_validation_new = data.iloc[validation_indices]
    
    return df_train_new, df_validation_new


def shuffle_instances(train, validation, test):
    """
    Shuffles the images to ensure that there are no dependencies between adjacent instances
    INPUT:
        train: Training dataframe (pandas.core.frame.DataFrame)
        validation: Validation dataframe (pandas.core.frame.DataFrame)
        test: Test dataframe (pandas.core.frame.DataFrame)
    OUTPUT:
        df_train: Shuffled training dataframe (pandas.core.frame.DataFrame)
        df_validation: Shuffled validation dataframe (pandas.core.frame.DataFrame)
        df_test: Shuffled test dataframe (pandas.core.frame.DataFrame)
    """
    
    df_train = train.sample(frac=1, random_state=42)
    df_validation = validation.sample(frac=1, random_state=42)
    df_test = test.sample(frac=1, random_state=42)
    
    return df_train, df_validation, df_test


def separate_features_labels(df_train, df_validation, df_test):
    """
    Separates (scaled) features and labels in each dataframe
    INPUT:
        df_train: Training dataframe (pandas.core.frame.DataFrame)
        df_validation: Validation dataframe (pandas.core.frame.DataFrame)
        df_test: Test dataframe (pandas.core.frame.DataFrame)
    OUTPUT:
        X_train: Scaled features of the training dataframe (pandas.core.frame.DataFrame)
        X_validation: Scaled features of the validation dataframe (pandas.core.frame.DataFrame)
        X_test: Scaled features of the test dataframe (pandas.core.frame.DataFrame)
        y_train: Labels of the training dataframe (pandas.core.series.Series)
        y_validation: Labels of the validation dataframe (pandas.core.series.Series)
        y_test: Labels of the test dataframe (pandas.core.series.Series)
    """
    
    X_train = df_train.iloc[:, 1:] / 255
    X_validation = df_validation.iloc[:, 1:] / 255
    X_test = df_test.iloc[:, 1:] / 255

    y_train = df_train.iloc[:, 0]
    y_validation = df_validation.iloc[:, 0]
    y_test = df_test.iloc[:, 0]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def pd_to_tf_dataset(X_train, X_validation, X_test, y_train, y_validation, y_test):
    """
    Creates TensorFlow datasets from Pandas dataframes
    INPUT:
        X_train: Scaled features of the training dataframe (pandas.core.frame.DataFrame)
        X_validation: Scaled features of the validation dataframe (pandas.core.frame.DataFrame)
        X_test: Scaled features of the test dataframe (pandas.core.frame.DataFrame)
        y_train: Labels of the training dataframe (pandas.core.series.Series)
        y_validation: Labels of the validation dataframe (pandas.core.series.Series)
        y_test: Labels of the test dataframe (pandas.core.series.Series)
    OUTPUT:
        train_dataset: TensorFlow training dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
        validation_dataset: TensorFlow validation dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
        test_dataset: TensorFlow test dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
    """
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, validation_dataset, test_dataset


def batch_dataset(train_dataset, train_batch, validation_dataset, validation_batch, test_dataset, test_batch):
    """
    Creates batch datasets from TensorFlow datasets
    INPUT:
        train_dataset: TensorFlow training dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
        train_batch: Amount of instances per training batch (int)
        validation_dataset: TensorFlow validation dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
        validation_batch: Amount of instances per validation batch (int)
        test_dataset: TensorFlow test dataset (tensorflow.python.data.ops.dataset_ops.TensorSliceDataset)
        test_batch: Amount of instances per test batch (int)
    OUTPUT:
        train_data: Batch training dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        validation_data: Batch validation dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        test_data: Batch test dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
    """
    
    train_data = train_dataset.batch(train_batch)
    validation_data = validation_dataset.batch(validation_batch)
    test_data = test_dataset.batch(test_batch)

    return train_data, validation_data, test_data


def prepare_early_stopping(patience):
    """
    Creates list of callbacks (early stopping)
    INPUT:
        patience: Tolerance of early stopping (int)
    OUTPUT:
        callbacks: List with the keras.callbacks.EarlyStopping object (list)
    """
    
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    callbacks = [es_callback]

    return callbacks


def reshape_dataset_for_mlp(train_data, validation_data, test_data):
    """
    Reshapes datasets so that they can be used as input for an MLP model
    INPUT:
        train_data: Batch training dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        validation_data: Batch validation dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        test_data: Batch test dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
    OUTPUT:
        train_data_mlp: Map training dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        validation_data_mlp: Map validation dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        test_data_mlp: Map test dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
    """
    
    train_data_mlp = train_data.map(lambda x, y: (tf.reshape(x, (-1, 784)), y))
    validation_data_mlp = validation_data.map(lambda x, y: (tf.reshape(x, (-1, 784)), y))
    test_data_mlp = test_data.map(lambda x, y: (tf.reshape(x, (-1, 784)), y))
    
    return train_data_mlp, validation_data_mlp, test_data_mlp


def reshape_dataset_for_gru(train_data, validation_data, test_data):
    """
    Reshapes datasets so that they can be used as input for a GRU model
    INPUT:
        train_data: Batch training dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        validation_data: Batch validation dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        test_data: Batch test dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
    OUTPUT:
        train_data_gru: Map training dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        validation_data_gru: Map validation dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        test_data_gru: Map test dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
    """
    
    train_data_gru = train_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28)), y))
    validation_data_gru = validation_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28)), y))
    test_data_gru = test_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28)), y))

    return train_data_gru, validation_data_gru, test_data_gru


def reshape_dataset_for_cnn(train_data, validation_data, test_data):
    """
    Reshapes datasets so that they can be used as input for a CNN model
    INPUT:
        train_data: Batch training dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        validation_data: Batch validation dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
        test_data: Batch test dataset (tensorflow.python.data.ops.dataset_ops.BatchDataset)
    OUTPUT:
        train_data_cnn: Map training dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        validation_data_cnn: Map validation dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        test_data_cnn: Map test dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
    """
    
    train_data_cnn = train_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28, 1)), y))
    validation_data_cnn = validation_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28, 1)), y))
    test_data_cnn = test_data.map(lambda x, y: (tf.reshape(x, (-1, 28, 28, 1)), y))
    
    return train_data_cnn, validation_data_cnn, test_data_cnn