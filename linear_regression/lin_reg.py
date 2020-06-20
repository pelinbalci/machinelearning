import os

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_synthetic_tf2-colab&hl=en#scrollTo=-mtVpoBrANAm


# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=None,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined create_model and train_model")


# title Define the plotting functions
# Plot the trained model against the training feature and label.
def plot_the_model(trained_weight, trained_bias, feature, label):
    # Label the axes.
    fig, ax = plt.subplots(1, sharex=True, figsize=(18, 15))
    ax.set_xlabel("feature")
    ax.set_ylabel("label")

    # Plot the feature values vs. label values.
    ax.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = my_feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    ax.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    fig_model = plt.gcf()

    return fig_model


def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""
    fig, ax = plt.subplots(1, sharex=True, figsize=(18, 15))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Root Mean Squared Error")

    ax.plot(epochs, rmse, label="Loss")
    ax.legend()
    ax.set_ylim([rmse.min()*0.97, rmse.max()])
    fig_loss = plt.gcf()

    return fig_loss


def save_plot(fig, name):
    file_name = name + '.png'
    path = os.path.join('/Users/pelin.balci/PycharmProjects/machinelearning/lin_reg_plots', file_name)
    fig.savefig(path)


print("Defined the plot_the_model and plot_the_loss_curve functions.")

my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

learning_rate = 0.1
epochs = 300
my_batch_size = 1


learning_rate_list = [0.1, 1, 2]
epochs_list = [10, 100, 300, 500]
batch_list = [1, 10, 300]


#for selecting the best batch_size, learning_rate and epochs:
for idx in range(len(batch_list)):
    my_batch_size = batch_list[idx]
    my_model = build_model(learning_rate)
    trained_weight, trained_bias, new_epochs_list, rmse = train_model(my_model, my_feature,
                                                             my_label, epochs,
                                                             my_batch_size)
    name_fig_model = 'model plot for eta: ' + str(learning_rate) + ',epochs: ' + str(epochs) + ' ,batch: ' + str(my_batch_size)
    name_fig_loss = 'loss plot for eta: ' + str(learning_rate) + ',epochs: ' + str(epochs) + ' ,batch: ' + str(
        my_batch_size)

    fig_model = plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    fig_loss = plot_the_loss_curve(new_epochs_list, rmse)

    save_plot(fig_model, name_fig_model)
    save_plot(fig_loss, name_fig_loss)
