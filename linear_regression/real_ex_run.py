import pandas as pd

from linear_regression.lin_reg_real_ex import import_dataset, build_model, train_model, plot_the_model, plot_the_loss_curve, save_plot, \
    predict_house_values

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
pd.set_option('display.max_columns', 10)

# training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_train.csv'

training_df = import_dataset(path)
# Scale the label.
training_df["median_house_value"] /= 1000.0
# Print the first rows of the pandas DataFrame.
print(training_df.head())
print(training_df.describe())

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 24
batch_size = 30

# Specify the feature and the label. Define a synthetic feature
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]

my_feature = "median_income"  # the total number of rooms on a specific city block.
my_label = "median_house_value"  # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based solely on total_rooms.

# Discard any pre-existing version of the model.
my_model = None


# The maximum value (max) of several columns seems very high compared to the other quantiles. For example,
# the total_rooms column. Given the quantile values (25%, 50%, and 75%), you might expect the max value of total_rooms
# to be approximately 5,000 or possibly 10,000. However, the max value is actually 37,937.

# When you see anomalies in a column, become more careful about using that column as a feature. That said, anomalies
# in potential features sometimes mirror anomalies in the label, which could make the column be (or seem to be)
# a powerful feature.
# Also, as you will see later in the course, you might be able to represent (pre-process) raw data in order to make
# columns into useful features.

# Invoke the functions.
my_model = build_model(learning_rate)
weight, bias, epochs_list, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

fig_model = plot_the_model(training_df, weight, bias, my_feature, my_label)
fig_loss = plot_the_loss_curve(epochs_list, rmse)

name_fig_model = 'real_model_plot_feature:' + str(my_feature) + ',eta: ' + str(learning_rate) + \
                 ',epochs: ' + str(epochs) + ',batch: ' + str(batch_size)
name_fig_loss = 'real_loss_plot_feature:' + str(my_feature) + ',eta: ' + str(learning_rate) + \
                 ',epochs: ' + str(epochs) + ',batch: ' + str(batch_size)

save_plot(fig_model, name_fig_model)
save_plot(fig_loss, name_fig_loss)

predict_house_values(training_df, my_model, 10, my_feature, my_label)

# Most of the predicted values differ significantly from the label value, so the trained model probably  doesn't have
# much predictive power. However, the first 10 examples might not be representative of  the rest of the examples.

# Training is not entirely deterministic, but population typically converges at a slightly higher RMSE than
# total_rooms.  So, population appears to be about the same or slightly worse at making predictions than total_rooms.

# Based on the loss values, this synthetic feature produces a better model than the individual features you tried in
# Task 2 and Task 3. However, the model still isn't creating great predictions.
