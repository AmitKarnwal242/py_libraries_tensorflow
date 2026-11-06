import tensorflow as tf
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
Y = np.array([0, 0, 0, 1, 1, 1])

for units in [8, 16, 32]:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, Y, epochs=50, verbose=0)

    loss, accuracy = model.evaluate(X, Y, verbose=0)

    print("Units:", units, "Accuracy:", accuracy)