import tensorflow as tf
import numpy as np

X = np.linspace(0, 2 * np.pi, 100)
Y = np.sin(X)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Training Started...")
model.fit(X, Y, epochs=500, verbose=0)
print("Training Completed!")

prediction = model.predict(np.array([[1.0]]), verbose=0)

print("Actual Sine Value:", np.sin(1.0))
print("Predicted Sine Value:", prediction[0][0])