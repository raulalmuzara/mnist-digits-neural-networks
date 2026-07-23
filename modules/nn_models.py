import os
import random
import tensorflow as tf


def MLP_model():
    """
    MultiLayer Perceptron model
    """
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(512, activation=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def GRU_model():
    """
    Gated Recurrent Unit model
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42), recurrent_initializer=tf.keras.initializers.Orthogonal(seed=42)),
        tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42), recurrent_initializer=tf.keras.initializers.Orthogonal(seed=42)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def CNN_model():
    """
    Convolutional Neural Network model
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=42), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=42), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=42), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=42), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model