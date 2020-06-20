from binary_classification.binary_classification_code import import_dataset, shuffle_df, normalize, \
    create_binary_label, create_feature_columns, create_feature_layer, create_model, train_model, plot_curve
# from binary_classification.param_binary_class import learning_rate, epochs, batch_size, label_name, METRICS,
# list_of_metrics_to_plot
# from parameters import learning_rate, epochs, batch_size, label_name, METRICS, list_of_metrics_to_plot
from binary_classification.param_auc import learning_rate, epochs, batch_size, label_name, METRICS, list_of_metrics_to_plot


test_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_test.csv'
train_path = '/Users/pelin.balci/PycharmProjects/machinelearning/inputs/california_housing_train.csv'

train_df = import_dataset(train_path)
test_df = import_dataset(test_path)

shuffled_train_df = shuffle_df(train_df)

train_df_norm = normalize(train_df)
test_df_norm = normalize(test_df)

train_df_norm, test_df_norm = create_binary_label(train_df_norm, test_df_norm, train_df, test_df)

feature_columns = create_feature_columns()
feature_layer = create_feature_layer(feature_columns)

model = create_model(learning_rate, feature_layer, METRICS)
epochs, hist = train_model(model, train_df_norm, epochs, batch_size, label_name, shuffle=True)

plot_curve(epochs, hist, list_of_metrics_to_plot)