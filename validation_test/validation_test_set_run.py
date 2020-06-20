import numpy as np

from validation_test.validation_test_set import import_dataset, scale_label, build_model, train_model, plot_the_loss_curve
from parameters import learning_rate, my_feature, my_label, batch_size, validation_split, epochs


# train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
# test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

test_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_test.csv'
train_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_train.csv'

train_df = import_dataset(train_path)
test_df = import_dataset(test_path)

train_df = scale_label(train_df)
test_df = scale_label(test_df)

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

# Invoke the functions to build and train the model.
my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, my_label, epochs, batch_size, validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], history["val_root_mean_squared_error"])


# Examine examples 0 through 4 and examples 25 through 29
# of the training set
train_df.head(n=1000)

# The original training set is sorted by longitude.
# Apparently, longitude influences the relationship of total_rooms to median_house_value.

# To fix the problem, shuffle the examples in the training set before splitting the examples into a training set and
# validation set. To do so, take the following steps:

# Shuffle the data in the training set by adding the following line anywhere before you call train_model
# (in the code cell associated with Task 1):

# Yes, after shuffling the original training set, the final loss for the training set and the
# validation set become much closer.

# If validation_split < 0.15, the final loss values for the training set and
# validation set diverge meaningfully.  Apparently, the validation set no longer contains enough examples.

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)

print(results)