'''Import relevant modules'''
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

'''Load Dataset'''
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data('/Users/pelin.balci/PycharmProjects/machinelearning/inputs/mnist.npz')


''' View the dataset '''
print(x_train[2917])

plt.imshow(x_train[2917])
#plt.show()

''' Normalize feature values '''
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0


''' View normalized dataset '''
print(x_train_normalized[2900][12]) # Output a normalized row

plt.imshow(x_train_normalized[2917])
#plt.show()


''' Define a plotting function '''
def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()


''' Create a deep neural net model '''
def create_model(my_learning_rate):
    """Create and compile a deep neural net."""

    # All models in this course are sequential.
    model = tf.keras.models.Sequential()

    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a a one-dimensional 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=32, activation='relu', name='Hidden1'))

    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Define the output layer.
    # The units parameter is set to 10 because the model must choose among 10 possible output values
    # (representing the digits from 0 to 9, inclusive).
    # Don't change this layer.
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


''' Invoke the previous functions'''
# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train, epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

print('make predictions')

probability_model = tf.keras.Sequential([my_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)
predictions[0]
np.argmax(predictions[0])
y_test[0]

# as is, number of nodes: 32 (1 layer) & reg_rate: 0.2
# loss: 0.1708 - accuracy: 0.9479 - val_loss: 0.1371 - val_accuracy: 0.9611
# test: loss: 0.1308 - accuracy: 0.9610

# number of nodes: 64 (1 layer) & reg_rate: 0.2
# loss: 0.0885 - accuracy: 0.9736 - val_loss: 0.0994 - val_accuracy: 0.9704
# test: loss: 0.0964 - accuracy: 0.9727

# number of nodes: 32 - 32 (2 layers) & reg_rate: 0.2
# loss: 0.1574 - accuracy: 0.9500 - val_loss: 0.1340 - val_accuracy: 0.9603
# test: loss: 0.1366 - accuracy: 0.9610

# number of nodes: 64 - 32 (2 layers) & reg_rate: 0.2
# loss: 0.0622 - accuracy: 0.9793 - val_loss: 0.0930 - val_accuracy: 0.9751
# test: loss: 0.0961 - accuracy: 0.9733

# number of nodes: 64 - 32 (2 layers) & reg rate: 0.4
# loss: 0.1379 - accuracy: 0.9556 - val_loss: 0.1090 - val_accuracy: 0.9693
# test: loss: 0.1094 - accuracy: 0.9690

# number of nodes: 64 - 64 (2 layers) & reg rate: 0.1
# loss: 0.0382 - accuracy: 0.9881 - val_loss: 0.0923 - val_accuracy: 0.9747
# test: loss: 0.0831 - accuracy: 0.9757

# number of nodes: 64 - 32 - 32 (3 layers) & reg rate: 0.2
# loss: 0.0539 - accuracy: 0.9821 - val_loss: 0.1008 - val_accuracy: 0.9734
# test: loss: 0.0972 - accuracy: 0.9723

# number of nodes: 256 (1 layer) & reg rate: 0.2
# loss: 0.0168 - accuracy: 0.9957 - val_loss: 0.0738 - val_accuracy: 0.9799
# test: loss: 0.0665 - accuracy: 0.9797

# When I change the activity function from softmax to sigmoid, it gives equal probability to each class (0.1).
