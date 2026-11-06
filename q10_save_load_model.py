import tensorflow as tf
import numpy as np

X = np.array([[1], [2], [3], [4]], dtype=float)
Y = np.array([[2], [4], [6], [8]], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse')
model.fit(X, Y, epochs=500, verbose=0)

# Save model
model.save("my_model.keras")
print("Model Saved Successfully")

# Load model
loaded_model = tf.keras.models.load_model("my_model.keras")
print("Model Loaded Successfully")

# Prediction
prediction = loaded_model.predict(np.array([[5.0]]), verbose=0)
print("Prediction for 5:", prediction[0][0])