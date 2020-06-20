# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/representation_with_a_feature_cross.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=representation_tf2-colab&hl=en#scrollTo=71WWwlhx4h0X

# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')


def import_dataset(path):
    df = pd.read_csv(path)
    return df


def scale_label(df):
    scale_factor = 1000.0
    # Scale the training set's label.
    df["median_house_value"] /= scale_factor
    return df


def shuffle_df(train_df):
    shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
    return shuffled_train_df


def create_feature_columns():
    # Create an empty list that will eventually hold all feature columns.
    feature_columns = []

    # Create a numerical feature column to represent latitude.
    latitude = tf.feature_column.numeric_column("latitude")
    feature_columns.append(latitude)

    longitude = tf.feature_column.numeric_column("longitude")
    feature_columns.append(longitude)
    return feature_columns


def create_feature_layer(feature_columns):
    # Convert the list of feature columns into a layer that will ultimately become
    # part of the model. Understanding layers is not important right now.
    fp_feature_layer = layers.DenseFeatures(feature_columns)
    return fp_feature_layer


def create_bucket_features(train_df, resolution_in_degrees):

    '''
    Each bin represents all the neighborhoods within a single degree.
    For example, neighborhoods at latitude 35.4 and 35.8 are in the same bucket,
    but neighborhoods in latitude 35.4 and 36.2 are in different buckets.

    The model will learn a separate weight for each bucket.
    For example, the model will learn one weight for all the neighborhoods in the "35" bin",
    a different weight for neighborhoods in the "36" bin, and so on.
    This representation will create approximately 20 buckets:

        10 buckets for latitude.
        10 buckets for longitude.
    '''

    # Create a new empty list that will eventually hold the generated feature column.
    feature_columns = []

    # Create a bucket feature column for latitude.
    latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
    latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                         int(max(train_df['latitude'])),
                                         resolution_in_degrees))
    latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                                   latitude_boundaries)
    feature_columns.append(latitude)

    # Create a bucket feature column for longitude.
    longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
    longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                          int(max(train_df['longitude'])),
                                          resolution_in_degrees))
    longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                    longitude_boundaries)
    feature_columns.append(longitude)

    # Convert the list of feature columns into a layer that will ultimately become
    # part of the model. Understanding layers is not important right now.
    buckets_feature_layer = layers.DenseFeatures(feature_columns)

    return feature_columns, buckets_feature_layer


def create_crossed_feature(feature_columns):
    latitude = feature_columns[0]
    longitude = feature_columns[1]

    # Create a feature cross of latitude and longitude.
    latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
    crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
    feature_columns.append(crossed_feature)

    # Convert the list of feature columns into a layer that will later be fed into
    # the model.
    feature_cross_feature_layer = layers.DenseFeatures(feature_columns)

    return feature_cross_feature_layer


# Define functions to create and train a model, and a plotting function
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
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    evaluation_result = model.evaluate(x=features, y=label, batch_size=batch_size)
    print('evaluation_result:', evaluation_result)

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


print("Defined the create_model, train_model, and plot_the_loss_curve functions.")


def evaluate_model(model, test_df, label_name, batch_size):
    print("\n: Evaluate the new model against the test set:")
    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))
    evaluation_test_result = model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
    print('evaluation_test_result:', evaluation_test_result)
    return evaluation_test_result

