import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras.datasets import boston_housing
import pandas as pd
import numpy as np

"""
This could did not work on my PC but worked in GC
"""

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(tf.__version__)
if isinstance(x_train, np.ndarray):
    print("data have been loaded as numpy array")

features = [' CRIM', 'ZN', 'INDUS', ' CHAS', ' NOX', ' RM', ' AGE', 'DIS', 'RAD', 'TAX',
            'PTRATIO', 'B', 'LSTAT']

x_train_df = pd.DataFrame(data=x_train, columns=features)
x_test_df = pd.DataFrame(data=x_test, columns=features)
y_train_df = pd.DataFrame(data=y_train, columns=['price'])
y_test_df = pd.DataFrame(data=y_test, columns=['price'])

print(x_train_df.head())

feature_columns = []
for feature_name in features:
    feature_columns.append(feature_column.numeric_column(feature_name, dtype=tf.float32))

"""
Have to create an input pipeline using tf.data
"""


def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds

    return input_function


train_input_fn = estimator_input_fn(x_train_df, y_train_df)
test_input_fn = estimator_input_fn(x_test_df, y_test_df, epochs=1, shuffle=False)


linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(test_input_fn)
print(result)

result = linear_est.predict(test_input_fn)
for pred, exp in zip(result, y_test[:32]):
    print("predicted Value: ", pred['predictions'][0], "Expected:  ", exp)

