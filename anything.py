import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('weatherAUS.csv')
Y, X = data['RainTomorrow'], data.drop(['RainTomorrow'], axis=1)

# Data cleaning for X

# mean
X['MinTemp'] = X['MinTemp'].apply(lambda x: X['MinTemp'].mean() if np.isnan(x) else x)
X['MaxTemp'] = X['MaxTemp'].apply(lambda x: X['MaxTemp'].mean() if np.isnan(x) else x)
X['Rainfall'] = X['Rainfall'].apply(lambda x: X['Rainfall'].mean() if np.isnan(x) else x)
X['WindGustSpeed'] = X['WindGustSpeed'].apply(lambda x: X['WindGustSpeed'].mean() if np.isnan(x) else x)
X['WindSpeed9am'] = X['WindSpeed9am'].apply(lambda x: X['WindSpeed9am'].mean() if np.isnan(x) else x)
X['WindSpeed3pm'] = X['WindSpeed3pm'].apply(lambda x: X['WindSpeed3pm'].mean() if np.isnan(x) else x)

# median
X['Humidity9am'] = X['Humidity9am'].apply(lambda x: X['Humidity9am'].median() if np.isnan(x) else x)
X['Humidity3pm'] = X['Humidity3pm'].apply(lambda x: X['Humidity3pm'].median() if np.isnan(x) else x)
X['Pressure9am'] = X['Pressure9am'].apply(lambda x: X['Pressure9am'].median() if np.isnan(x) else x)
X['Pressure3pm'] = X['Pressure3pm'].apply(lambda x: X['Pressure3pm'].median() if np.isnan(x) else x)
X['Cloud9am'] = X['Cloud9am'].apply(lambda x: X['Cloud9am'].median() if np.isnan(x) else x)
X['Cloud3pm'] = X['Cloud3pm'].apply(lambda x: X['Cloud3pm'].median() if np.isnan(x) else x)
X['Temp9am'] = X['Temp9am'].apply(lambda x: X['Temp9am'].median() if np.isnan(x) else x)
X['Temp3pm'] = X['Temp3pm'].apply(lambda x: X['Temp3pm'].median() if np.isnan(x) else x)

# drop
X.drop(['Evaporation', 'Sunshine'], axis=1, inplace=True)

# binary
X['RainToday'] = X['RainToday'].apply(lambda x: 1.0 if x == 'Yes' else 0.0)
X['RainToday'].fillna(0.0, inplace=True)  # Fill missing binary values with 0.0 (No)

# filling na values in wind direction with mode
X['WindGustDir'].fillna(X['WindGustDir'].mode()[0], inplace=True)
X['WindDir9am'].fillna(X['WindDir9am'].mode()[0], inplace=True)
X['WindDir3pm'].fillna(X['WindDir3pm'].mode()[0], inplace=True)

# making separate column for each unique direction in the WindGustDir column
X = pd.get_dummies(X, columns=['WindGustDir'])

# making separate column for each unique direction in the WindDir9am column
X = pd.get_dummies(X, columns=['WindDir9am'])

# making separate column for each unique direction in the WindDir3pm column
X = pd.get_dummies(X, columns=['WindDir3pm'])

# drop the original wind direction columns after one-hot encoding
X.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1, inplace=True)

# date column
X['Date'] = pd.to_datetime(X['Date'])
X.dropna(subset=['Date'], inplace=True)

# location
X = pd.get_dummies(X, columns=['Location'])

# drop the original 'Location' column after one-hot encoding
X.drop(['Location'], axis=1, inplace=True)

X.info()

# Data cleaning for Y
Y = Y.apply(lambda x: 1.0 if x == 'Yes' else 0.0)
Y.fillna(0.0, inplace=True)  # Fill missing binary values with 0.0 (No)

Y.info()

