# Original File: https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_ml_fairness.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=fairness_tf2-colab&hl=en#scrollTo=MW-qryqs1gig

#/Users/pelin.balci/PycharmProjects/machinelearning/inputs/facets_overview-1.0.0-py2.py3-none-any.whl

'''
Prediction Task
The prediction task is to determine whether a person makes over $50,000 US Dollar a year.

Label
income_bracket: Whether the person makes more than $50,000 US Dollars annually.
'''

#@title Import revelant modules and install Facets
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

train_csv_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/adult_census_train.csv'
test_csv_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/adult_census_test.csv'

train_df = pd.read_csv(train_csv_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
test_df = pd.read_csv(test_csv_path, names=COLUMNS, sep=r'\s*,\s*', skiprows=[0], engine='python', na_values="?")


def pandas_to_np(df):
    df = df.dropna(how="any", axis=0)

    labels = np.array(df['income_bracket'] == ">50K")
    features = df.drop('income_bracket', axis=1)
    features = {name: np.array(value) for name, value in features.items()}

    return features, labels


age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
race = tf.feature_column.categorical_column_with_vocabulary_list("race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(age_buckets),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
]

#deep_columns.append(capital_gain)
#deep_columns.append(hours_per_week)


# Parameters from form fill-ins
HIDDEN_UNITS_LAYER_01 = 128
HIDDEN_UNITS_LAYER_02 = 64
LEARNING_RATE = 0.1
L1_REGULARIZATION_STRENGTH = 0.001  # @param
L2_REGULARIZATION_STRENGTH = 0.001  # @param

RANDOM_SEED = 512
tf.random.set_seed(RANDOM_SEED)

# List of built-in metrics that we'll need to evaluate performance.
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]


EPOCHS = 10
BATCH_SIZE = 500

features, labels = pandas_to_np(train_df)

regularizer = tf.keras.regularizers.l1_l2(l1=L1_REGULARIZATION_STRENGTH, l2=L2_REGULARIZATION_STRENGTH)

model = tf.keras.Sequential([
    layers.DenseFeatures(deep_columns),
    layers.Dense(HIDDEN_UNITS_LAYER_01, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(HIDDEN_UNITS_LAYER_02, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
])

model.compile(optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

model.fit(x=features, y=labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

print('done')

features, labels = pandas_to_np(test_df)
model.evaluate(x=features, y=labels)


# as is:
# loss: 0.4291 - tp: 4039.0000 - fp: 1665.0000 - tn: 20989.0000 - fn: 3469.0000 - accuracy: 0.8298 - precision: 0.7081 - recall: 0.5380 - auc: 0.8796
# Tloss: loss: 0.4463 - tp: 0 - fp: 3097.0000 - tn: 11963.0000 - fn: 0.0000e+00 - accuracy: 0.7944 - precision: 0.000 - recall: 0.000 - auc: 0.0000
