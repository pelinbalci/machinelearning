# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

import numpy as np
import tensorflow as tf

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

learning_rate = 0.1
define_layers = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([define_layers])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print(model.predict([100]))
