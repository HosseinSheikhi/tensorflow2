import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow import feature_column as fc
import pandas as pd

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT']
x_train_df = pd.DataFrame(x_train, columns=features)
x_test_df = pd.DataFrame(x_test, columns=features)
y_train_df = pd.DataFrame(y_train, columns=['price'])
y_test_df = pd.DataFrame(y_test, columns=['price'])

x_train_df.head()

feature_columns = []

for feature_name in features:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

"""
create the input function for the estimator
the function returns the tf.Data.Dataset object with a tuple: features and labels in batches 
"""


def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=10):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds

    return input_function


train_input_function = estimator_input_fn(x_train_df, y_train_df)
val_input_function = estimator_input_fn(x_test_df, y_test_df, epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_est.train(train_input_function, steps=100)
result = linear_est.evaluate(val_input_function)

result = linear_est.predict(val_input_function)
for pred, exp in zip(result, y_test[:32]):
    print("predicted value: {}   and Expected value: {}".format(pred['prediction'][0], exp))
