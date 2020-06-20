import tensorflow as tf

# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = 'median_house_value_is_high'

resolution_in_degrees = 0.4

classification_threshold = 0.35

# Establish the metrics the model will measure.
METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                           threshold=classification_threshold),]

list_of_metrics_to_plot = ['accuracy']

# Split the original training set into a reduced training set and a validation set.
validation_split = 0.2

# Identify the feature and the label.
my_feature = "median_income"  # the median income on a specific city block.
my_label = "median_house_value"  # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based  solely on the neighborhood's median income.

# Discard any pre-existing version of the model.
my_model = None