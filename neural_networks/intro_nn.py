# Original File:
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=intro_to_nn_tf2-colab&hl=en

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


def import_dataset(path):
    df = pd.read_csv(path)
    return df


def shuffle_df(train_df):
    shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
    return shuffled_train_df


def normalize(df):
    '''
    Calculate the Z-scores of each column in the training set and
    write those Z-scores into a new pandas DataFrame named train_df_norm.

    Examine some of the values of the normalized training set. Notice that most
    Z-scores fall between -2 and +2.
    '''
    df_mean = df.mean()
    df_std = df.std()
    df_norm = (df - df_mean) / df_std
    return df_norm


def create_feature_columns(train_df_norm):
    # Create an empty list that will eventually hold all feature columns.
    feature_columns = []

    # We scaled all the columns, including latitude and longitude, into their
    # Z scores. So, instead of picking a resolution in degrees, we're going
    # to use resolution_in_Zs.  A resolution_in_Zs of 1 corresponds to
    # a full standard deviation.
    resolution_in_Zs = 0.3  # 3/10 of a standard deviation.

    # Represent median_income as a floating-point value.
    median_income = tf.feature_column.numeric_column("median_income")
    feature_columns.append(median_income)

    # Represent population as a floating-point value.
    population = tf.feature_column.numeric_column("population")
    feature_columns.append(population)

    # Create a bucket feature column for latitude. ex: [-1, -0.7, -0.4, -0.1, 0.2 ...]
    latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
    latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                         int(max(train_df_norm['latitude'])),
                                         resolution_in_Zs))
    latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

    # Create a bucket feature column for longitude.
    longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
    longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                          int(max(train_df_norm['longitude'])),
                                          resolution_in_Zs))
    longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                    longitude_boundaries)

    # Create a feature cross of latitude and longitude.
    latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
    crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
    feature_columns.append(crossed_feature)

    return feature_columns


def create_feature_layers(feature_columns):
    # Convert the list of feature columns into a layer that will later be fed into
    # the model.
    my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    return my_feature_layer


def plot_the_loss_curve(epochs, mse):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max() * 1.03])
    plt.show()


# Define functions to create and train a linear regression model
def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def create_deep_neural_net_model(my_learning_rate, my_feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(my_feature_layer)

    # Describe the topography of the model by calling the tf.keras.layers.Dense
    # method once for each layer. We've specified the following arguments:
    #   * units specifies the number of nodes in this layer.
    #   * activation specifies the activation function (Rectified Linear Unit).
    #   * name is just a string that can be useful when debugging.

    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=20,
                                    activation='relu',
                                    #kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden1'))

    #model.add(tf.keras.layers.Dropout(rate=0.25))

    # Define the second hidden layer with 12 nodes.
    model.add(tf.keras.layers.Dense(units=12,
                                    activation='relu',
                                    #kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden2'))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""

    # Split the dataset into features and label.
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True)

    # Get details that will be useful for plotting the loss curve.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse







