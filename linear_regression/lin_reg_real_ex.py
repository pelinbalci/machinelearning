# Copyright 2020 Google LLC. Double-click here for license information.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_real_tf2-colab&hl=en#scrollTo=zFGKL45LO8Tt

import os

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
pd.set_option('display.max_columns',10)


def import_dataset(path):
    training_df = pd.read_csv(path)
    return training_df


# Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined the create_model and traing_model functions.")


# Define the plotting functions
def plot_the_model(training_df, trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples."""

    fig, ax = plt.subplots(1, sharex=True, figsize=(18, 15))

    # Label the axes.plt.xlabel(feature)
    ax.set_ylabel(label)

    # Create a scatter plot from 200 random points of the dataset.
    random_examples = training_df.sample(n=200)
    ax.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    ax.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    fig_model = plt.gcf()

    return fig_model


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""
    fig, ax = plt.subplots(1, sharex=True, figsize=(18, 15))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Root Mean Squared Error")

    ax.plot(epochs, rmse, label="Loss")
    ax.legend()
    ax.set_ylim([rmse.min()*0.97, rmse.max()])
    fig_loss = plt.gcf()

    return fig_loss


print("Defined the plot_the_model and plot_the_loss_curve functions.")


def predict_house_values(training_df, my_model, n, feature, label):
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][i],
                                       training_df[label][i],
                                       predicted_values[i][0]))


def save_plot(fig, name):
    file_name = name + '.png'
    path = os.path.join('/Users/pelin.balci/PycharmProjects/machinelearning/real_example_plots', file_name)
    fig.savefig(path)



