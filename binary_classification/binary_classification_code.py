# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/binary_classification.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=binary_classification_tf2-colab&hl=en

'''
 In this exercise, the binary question will be, "Are houses in this neighborhood above a certain price?"

After doing this Colab, you'll know how to:

Convert a regression question into a classification question.
Modify the classification threshold and determine how that modification influences the model.
Experiment with different classification metrics to determine your model's effectiveness.
'''

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
# tf.keras.backend.set_floatx('float32')


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


def create_binary_label(train_df_norm, test_df_norm, train_df, test_df):
    threshold = 265000  # This is the 75th percentile for median house values.
    train_df_norm["median_house_value_is_high"] = (train_df['median_house_value'] > threshold).astype(float)
    test_df_norm["median_house_value_is_high"] = (test_df['median_house_value'] > threshold).astype(float)

    return train_df_norm, test_df_norm


def create_feature_columns():
    # Create an empty list that will eventually hold all feature columns.
    feature_columns = []

    # Create a numerical feature column to represent latitude.
    median_income = tf.feature_column.numeric_column("median_income")
    feature_columns.append(median_income)

    total_rooms = tf.feature_column.numeric_column("total_rooms")
    feature_columns.append(total_rooms)
    return feature_columns


def create_feature_layer(feature_columns):
    # Convert the list of feature columns into a layer that will later be fed into
    # the model.
    feature_layer = layers.DenseFeatures(feature_columns)

    # Print the first 3 and last 3 rows of the feature_layer's output when applied
    # to train_df_norm:
    #print(feature_layer(dict(train_df_norm)))

    return feature_layer


def create_model(learning_rate, feature_layer, metrics):
    model = tf.keras.models.Sequential()

    model.add(feature_layer)

    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    return model


def train_model(model, train_df_norm, epochs, batch_size, label_name, shuffle=True):
    features = {name: np.array(value) for name, value in train_df_norm.items()}
    label = np.array(features.pop(label_name))

    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=shuffle)

    epochs = history.epoch
    loss_values = history.history
    hist = pd.DataFrame(loss_values)

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # ex: list_of_metrics_to_plot = ['accuracy']
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics: # list od metrics elements are loss metrics.
        x = hist[m] # hist is a dataframe, contains loss values: loss and accuracy.
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()