# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

import numpy as np
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

'''
Build a layer
We'll call the layer l0 and create it by instantiating tf.keras.layers.Dense with the following configuration:

input_shape=[1] — This specifies that the input to this layer is a single value.
That is, the shape is a one-dimensional array with one member.
Since this is the first (and only) layer, that input shape is the input shape of the entire model.
The single value is a floating point number, representing degrees Celsius.

units=1 — This specifies the number of neurons in the layer.
The number of neurons defines how many internal variables the layer has to try to learn how to solve the problem
(more later). Since this is the final layer, it is also the size of the model's output — a single float value
representing degrees Fahrenheit. (In a multi-layered network, the size and shape of the layer would need to match the
input_shape of the next layer.)
'''
learning_rate = 0.1
l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) # or you can use model.add(l0)

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

'''
The first argument is the inputs, the second argument is the desired outputs.
The epochs argument specifies how many times this cycle should be run, and
the verbose argument controls how much output the method produces.
'''

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100.0]))

print("These are the layer variables: {}".format(l0.get_weights()))

'''
This is really close to the values in the conversion formula. We'll explain this in an upcoming video where we show how a Dense layer works, but for a single neuron with a single input and a single output, the internal math looks the same as the equation for a line,
𝑦=𝑚𝑥+𝑏 , which has the same form as the conversion equation,  𝑓=1.8𝑐+32 .
'''


l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))
