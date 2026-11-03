import tensorflow as tf
import numpy as np

# Custom loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Training data
X = np.array([1, 2, 3, 4], dtype=float)
Y = np.array([2, 4, 6, 8], dtype=float)

# Create model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile model using custom loss
model.compile(optimizer='sgd', loss=custom_loss)

# Train model
model.fit(X, Y, epochs=100, verbose=0)

print("Custom Loss Function Used Successfully")
print("Final Loss:", model.evaluate(X, Y, verbose=0))