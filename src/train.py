import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, save_path=None):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = []
    if save_path:
        # ensure directory exists only when a directory part is present
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        ckpt = ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max')
        callbacks.append(ckpt)

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    # no extra save needed because ModelCheckpoint already saves the best model

def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--save', type=str, default=None, help='Path to save best model (e.g. models/cifar10.h5)')
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    print(f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}, save={args.save}")
    train_model(model, x_train, y_train, x_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size, save_path=args.save)

if __name__ == "__main__":
    main()