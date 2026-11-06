import tensorflow as tf
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
Y = np.array([[2], [4], [6], [8], [10]], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse')

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=10
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor='loss',
    save_best_only=True
)

model.fit(
    X, Y,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    verbose=0
)

print("Training Completed")
print("Best Model Saved Successfully")