import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import re
import matplotlib.pyplot as plt


boston = pd.read_csv('listings.csv')
used_features = ['property_type','room_type','bathrooms','bedrooms','beds','bed_type','accommodates','host_total_listings_count','number_of_reviews','review_scores_value','neighbourhood_cleansed','cleaning_fee','minimum_nights','security_deposit','host_is_superhost', 'instant_bookable', 'price']
boston = boston.loc[:, used_features]
boston.head(2)

for feature in ['cleaning_fee', 'security_deposit', 'price']:
    boston[feature] = boston[feature].map(lambda x: x.replace("$", "").replace(",", ""), na_action = 'ignore')
    boston[feature] = boston[feature].astype(float)
    boston[feature].fillna(boston[feature].median(), inplace = True)


for feature in ["bathrooms","bedrooms","beds","review_scores_value"]:
    boston[feature].fillna(boston[feature].median(),inplace = True)


boston['property_type'].fillna('Apartment',inplace = True)


# boston['price'].plot(kind = 'hist', grid = True)
# plt.show()


boston['price'] = np.log(boston['price'])

features = boston.drop('price', axis = 1)
target = boston['price']

# split data into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# feature column
numeric_columns = ['host_total_listings_count','accommodates','bathrooms','bedrooms','beds','security_deposit','cleaning_fee','minimum_nights','number_of_reviews','review_scores_value']
categorical_columns = ['host_is_superhost','neighbourhood_cleansed','property_type','room_type','bed_type','instant_bookable']
numeric_features = [tf.feature_column.numeric_column(key = column) for column in numeric_columns]
categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(key = column, vocabulary_list=features[column].unique()) for column in categorical_columns]
linear_features =  numeric_features + categorical_features


training_input_fn = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 32, shuffle = True, num_epochs= None)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size= 32, shuffle = False, num_epochs= 1)
linear_regressor = tf.estimator.LinearRegressor(feature_columns= linear_features, model_dir = 'linear_regressor')
linear_regressor.train(input_fn = training_input_fn,steps=10000)
linear_regressor.evaluate(input_fn = eval_input_fn)
list(linear_regressor.predict(input_fn = eval_input_fn))
