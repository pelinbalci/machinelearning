import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", 
           "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]


''' read files'''    
train_csv_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/adult_census_train.csv'
test_csv_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/adult_census_test.csv'

train_df = pd.read_csv(train_csv_path, names=COLUMNS, sep=r'\s*,\s*', engine='python', na_values="?")
test_df = pd.read_csv(test_csv_path, names=COLUMNS, sep=r'\s*,\s*', skiprows=[0], engine='python', na_values="?")


''' remove missing values ''' 
train_df = train_df.dropna(how="any", axis=0)
test_df = test_df.dropna(how='any', axis=0)


''' extract features and labels '''
def labels_features(df):
    labels = np.array(df['income_bracket'] == ">50K")
    features = df.drop('income_bracket', axis=1)
    features = {name: np.array(value) for name, value in features.items()}
    return labels, features

train_labels, train_features = labels_features(train_df)
test_labels, test_features = labels_features(test_df)


''' numeric columns '''
age = tf.feature_column.numeric_column('age')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

''' numeric columns to buckets '''
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

''' categoric column '''
workclass_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

workclass = tf.feature_column.indicator_column(workclass_categorical)

education_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
education = tf.feature_column.indicator_column(education_categorical)


relationship_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])

relationship = tf.feature_column.indicator_column(relationship_categorical)

''' categoric columns - many different values'''
native_country_categorical = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)
native_country = tf.feature_column.embedding_column(native_country_categorical, dimension=8)

occupation_categorical = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
occupation = tf.feature_column.embedding_column(occupation_categorical, dimension=8)


''' create a feature column'''
deep_columns = []
deep_columns.append(age_buckets)
deep_columns.append(workclass)
deep_columns.append(education)
deep_columns.append(relationship)
deep_columns.append(native_country)
deep_columns.append(occupation)

''' define randomness '''
RANDOM_SEED = 512
tf.random.set_seed(RANDOM_SEED)

''' create feature_layer '''
feature_layer = tf.keras.layers.DenseFeatures(deep_columns)

''' define regularizer '''
regularizer = tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)


''' define model '''
model = tf.keras.models.Sequential()

''' add layers '''
model.add(feature_layer)

model.add(tf.keras.layers.Dense(units=128,
                                activation='relu',
                                kernel_regularizer=regularizer,
                                name='Hidden1'))

model.add(tf.keras.layers.Dense(units=64,
                                activation='relu',
                                kernel_regularizer=regularizer,
                                name='Hidden1'))

model.add(tf.keras.layers.Dense(1,
                                activation='sigmoid',
                                kernel_regularizer=regularizer))

''' define evaluation metrics '''
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

''' compile the model '''
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

''' fit the model train features and labels '''
model.fit(x=train_features, y=train_labels, epochs=10, batch_size=500)

''' evaluate the results '''
test_results = model.evaluate(x=test_features, y=test_labels, verbose=0)
confusion_matrix = np.array([[test_results[1], test_results[4]],
                             [test_results[2], test_results[3]]])

test_performance_metrics = {
    'ACCURACY': test_results[5],
    'PRECISION': test_results[6],
    'RECALL': test_results[7],
    'AUC': test_results[8]
}

print(confusion_matrix)
print(test_performance_metrics)


