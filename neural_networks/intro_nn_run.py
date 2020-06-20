# Original File:
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=intro_to_nn_tf2-colab&hl=en

import numpy as np

from neural_networks.intro_nn import \
    import_dataset, \
    shuffle_df, \
    normalize, \
    create_feature_columns, \
    create_feature_layers, \
    plot_the_loss_curve, \
    create_model, \
    create_deep_neural_net_model, \
    train_model
from neural_networks.nn_params import learning_rate, epochs, batch_size, label_name

# Prepare Data:
test_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_test.csv'
train_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_train.csv'

train_df = import_dataset(train_path)
test_df = import_dataset(test_path)

shuffled_train_df = shuffle_df(train_df)

train_df_norm = normalize(train_df)
test_df_norm = normalize(test_df)


# Establish the model's topography.
feature_columns = create_feature_columns(train_df_norm)
my_feature_layer = create_feature_layers(feature_columns)

my_model = create_model(learning_rate, my_feature_layer)
my_deep_model = create_deep_neural_net_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set.
epochs_list, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs_list, mse)

# Train the model on the normalized training set.
epochs_deep, mse_deep = train_model(my_deep_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs_deep, mse_deep)

test_features = {name: np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label

print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

print("\n Evaluate the deep neural network model against the test set:")
my_deep_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

''' Comparison Results'''
# In our experiments, the loss of the deep neural network model was consistently lower than
# that of the linear regression model, which suggests that the deep neural network model
# will make better predictions than the linear regression model.

'''How many nodes are enough?'''
# Setting the topography as follows produced reasonably good results with relatively few nodes:
#       * 10 nodes in the first layer.
#       *  6 nodes in the second layer.
# As the number of nodes in each layer dropped below the preceding, test loss increased.
# However, depending on your application, hardware constraints, and the relative pain inflicted
# by a less accurate model, a smaller network (for example, 6 nodes in the first layer and 4 nodes in the second layer)
# might be acceptable.

'''Regularization'''
# Notice that the model's loss against the test set is much higher than the loss against the training set.
# In other words, the deep neural network is overfitting to the data in the training set.
# To reduce overfitting, regularize the model.

'''  When you add a regularization function to a model, you might need to tweak other hyperparameters. '''

