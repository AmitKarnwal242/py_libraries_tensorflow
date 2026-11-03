import tensorflow as tf
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# Compile model
model.compile(optimizer='sgd', loss='mse')

# Train model
model.fit(X, Y, epochs=500, verbose=0)

# Predict output
result = model.predict(np.array([6.0]), verbose=0)

print("Prediction for 6:", result[0][0])