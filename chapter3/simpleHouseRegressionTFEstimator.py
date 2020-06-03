"""
We have to features "the area' and "its type (bungalow or apartment)

FEATURE COLUMNS:
The input parameters to be used by the estimator for training are passed as feature columns

"""

import tensorflow as tf
from tensorflow import feature_column as fc

numeric_column = fc.numeric_column  # for area
categorical_column_with_vocabulary_list = fc.categorical_column_with_vocabulary_list  # for house type

feature = [tf.feature_column.numeric_column('area'),
           tf.feature_column.categorical_column_with_vocabulary_list('type', ['bungalow', 'house'])]


def train_input_fn():
    features = {
        "area": [1000, 2000, 4000, 1000, 2000, 4000],
        "type": ['bungalow', 'bungalow', 'bungalow', 'house', 'house', 'house']
    }
    labels = [500, 1000, 1500, 700, 1300, 1900]
    return features, labels


model = tf.estimator.LinearRegressor(feature)
model.train(train_input_fn, steps=200)


def predict_input_function():
    features = {
        "area": [1500, 1800],
        "type": ["bungalow", "house"]
    }
    return features


predictions = model.predict(predict_input_function)
print(next(predictions))
print(next(predictions))
