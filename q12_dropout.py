import tensorflow as tf
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
Y = np.array([0, 0, 0, 1, 1, 1])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, Y, epochs=100, verbose=0)

print("Model Training Completed")
print("Dropout Layers Added Successfully")