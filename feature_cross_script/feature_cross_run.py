from feature_cross_script.feature_cross import import_dataset, scale_label, shuffle_df, create_model, train_model, \
    plot_the_loss_curve, create_feature_columns, create_feature_layer, create_bucket_features, create_crossed_feature,\
    evaluate_model
from parameters import learning_rate, epochs, batch_size, label_name, resolution_in_degrees


test_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_test.csv'
train_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_train.csv'

train_df = import_dataset(train_path)
test_df = import_dataset(test_path)

train_df = scale_label(train_df)
test_df = scale_label(test_df)

shuffled_train_df = shuffle_df(train_df)

# Use floating latitude ang longitude vectors seperately:
feature_columns = create_feature_columns()
fp_feature_layer = create_feature_layer(feature_columns)

# Bucketize them intp 10 integer points, we still have two separate vectors:
feature_columns, buckets_feature_layer = create_bucket_features(train_df, resolution_in_degrees)
''' Bucket representation outperformed floating-point representations.  '''

# In real life we have two dimension vectors for latitude and longitude, cross them:
feature_cross_feature_layer = create_crossed_feature(feature_columns)
''' Representing these features as a feature cross produced much lower loss values than 
representing these features as buckets'''

# Create and compile the model's topography.
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

evaluation_test_result = evaluate_model(my_model, test_df, label_name, batch_size)

print('done')
