# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt

# data = pd.read_csv('weatherAUS.csv')
# Y, X = data['RainTomorrow'], data.drop(['RainTomorrow'], axis=1)

# # data cleaning for X

# # mean
# X['MinTemp']=X['MinTemp'].apply(lambda x:X['MinTemp'].mean() if np.isnan(x) else x)
# X['MaxTemp']=X['MaxTemp'].apply(lambda x:X['MaxTemp'].mean() if np.isnan(x) else x)
# X['Rainfall']=X['Rainfall'].apply(lambda x:X['Rainfall'].mean() if np.isnan(x) else x)
# X['WindGustSpeed']=X['WindGustSpeed'].apply(lambda x:X['WindGustSpeed'].mean() if np.isnan(x) else x)
# X['WindSpeed9am']=X['WindSpeed9am'].apply(lambda x:X['WindSpeed9am'].mean() if np.isnan(x) else x)
# X['WindSpeed3pm']=X['WindSpeed3pm'].apply(lambda x:X['WindSpeed3pm'].mean() if np.isnan(x) else x)

# # median
# X['Humidity9am']=X['Humidity9am'].apply(lambda x:X['Humidity9am'].median() if np.isnan(x) else x)
# X['Humidity3pm']=X['Humidity3pm'].apply(lambda x:X['Humidity3pm'].median() if np.isnan(x) else x)
# X['Pressure9am']=X['Pressure9am'].apply(lambda x:X['Pressure9am'].median() if np.isnan(x) else x)
# X['Pressure3pm']=X['Pressure3pm'].apply(lambda x:X['Pressure3pm'].median() if np.isnan(x) else x)
# X['Cloud9am']=X['Cloud9am'].apply(lambda x:X['Cloud9am'].median() if np.isnan(x) else x)
# X['Cloud3pm']=X['Cloud3pm'].apply(lambda x:X['Cloud3pm'].median() if np.isnan(x) else x)
# X['Temp9am']=X['Temp9am'].apply(lambda x:X['Temp9am'].median() if np.isnan(x) else x)
# X['Temp3pm']=X['Temp3pm'].apply(lambda x:X['Temp3pm'].median() if np.isnan(x) else x)

# # drop
# X.drop(['Evaporation'],axis=1,inplace=True)
# X.drop(['Sunshine'],axis=1,inplace=True)

# # binary
# X['RainToday']=X['RainToday'].apply(lambda x: 1.0 if x=='Yes' else 0.0)
# X['RainToday']=X['RainToday'].apply(lambda x: X['RainToday'].mode() if np.isnan(x) else x)

# # filling na values in wind direction and mode is S for all three
# X['WindGustDir'].fillna(X['WindGustDir'].mode()[0], inplace=True)
# X['WindDir9am'].fillna(X['WindDir9am'].mode()[0], inplace=True)
# X['WindDir3pm'].fillna(X['WindDir3pm'].mode()[0], inplace=True)

# #making seperate column for each unique direction in the WindGustDir column
# X=pd.get_dummies(X,columns=['Wind Gust Dir'])
# X['WindGustDir_E']=X['WindGustDir_E'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_ENE']=X['WindGustDir_ENE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_ESE']=X['WindGustDir_ESE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_N']=X['WindGustDir_N'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_NE']=X['WindGustDir_NE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_NNE']=X['WindGustDir_NNE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_NNW']=X['WindGustDir_NNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_NW']=X['WindGustDir_NW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_S']=X['WindGustDir_S'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_SE']=X['WindGustDir_SE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_SSE']=X['WindGustDir_SSE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_SSW']=X['WindGustDir_SSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_SW']=X['WindGustDir_SW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_W']=X['WindGustDir_W'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_WNW']=X['WindGustDir_WNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindGustDir_WSW']=X['WindGustDir_WSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X=X.drop(['Wind Gust Dir'],axis=1)

# #making seperate column for each unique direction in the WindDir9am column
# X=pd.get_dummies(X,columns=['WindDir9am'])
# X['WindDir9am_E']=X['WindDir9am_E'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_ENE']=X['WindDir9am_ENE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_ESE']=X['WindDir9am_ESE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_N']=X['WindDir9am_N'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_NE']=X['WindDir9am_NE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_NNE']=X['WindDir9am_NNE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_NNW']=X['WindDir9am_NNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_NW']=X['WindDir9am_NW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_S']=X['WindDir9am_S'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_SE']=X['WindDir9am_SE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_SSE']=X['WindDir9am_SSE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_SSW']=X['WindDir9am_SSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_SW']=X['WindDir9am_SW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_W']=X['WindDir9am_W'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_WNW']=X['WindDir9am_WNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir9am_WSW']=X['WindDir9am_WSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X=X.drop(['WindDir9am'],axis=1)

# #making seperate column for each unique direction in the WindDir3pm column
# X=pd.get_dummies(X,columns=['WindDir3pm'])
# X['WindDir3pm_E']=X['WindDir3pm_E'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_ENE']=X['WindDir3pm_ENE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_ESE']=X['WindDir3pm_ESE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_N']=X['WindDir3pm_N'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_NE']=X['WindDir3pm_NE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_NNE']=X['WindDir3pm_NNE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_NNW']=X['WindDir3pm_NNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_NW']=X['WindDir3pm_NW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_S']=X['WindDir3pm_S'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_SE']=X['WindDir3pm_SE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_SSE']=X['WindDir3pm_SSE'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_SSW']=X['WindDir3pm_SSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_SW']=X['WindDir3pm_SW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_W']=X['WindDir3pm_W'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_WNW']=X['WindDir3pm_WNW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['WindDir3pm_WSW']=X['WindDir3pm_WSW'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X=X.drop(['WindDir3pm'],axis=1)

# #date column
# X['Date']=pd.to_datetime(X['Date'])
# X.dropna(subset=['Date'],inplace=True)

# #location
# X=pd.get_dummies(X,columns=['Location'])
# X['Location_Adelaide']=X['Location_Adelaide'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Albany']=X['Location_Albany'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Albury']=X['Location_Albury'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_AliceSprings']=X['Location_AliceSprings'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_BadgerysCreek']=X['Location_BadgerysCreek'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Ballarat']=X['Location_Ballarat'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Bendigo']=X['Location_Bendigo'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Brisbane']=X['Location_Brisbane'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Cairns']=X['Location_Cairns'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Canberra']=X['Location_Canberra'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Cobar']=X['Location_Cobar'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_CoffsHarbour']=X['Location_CoffsHarbour'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Dartmoor']=X['Location_Dartmoor'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Darwin']=X['Location_Darwin'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_GoldCoast']=X['Location_GoldCoast'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Hobart']=X['Location_Hobart'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Katherine']=X['Location_Katherine'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Launceston']=X['Location_Launceston'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Melbourne']=X['Location_Melbourne'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_MelbourneAirport']=X['Location_MelbourneAirport'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Mildura']=X['Location_Mildura'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Moree']=X['Location_Moree'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_MountGambier']=X['Location_MountGambier'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_MountGinini']=X['Location_MountGinini'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Newcastle']=X['Location_Newcastle'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Nhil']=X['Location_Nhil'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_NorahHead']=X['Location_NorahHead'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_NorfolkIsland']=X['Location_NorfolkIsland'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Nuriootpa']=X['Location_Nuriootpa'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_PearceRAAF']=X['Location_PearceRAAF'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Penrith']=X['Location_Penrith'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Perth']=X['Location_Perth'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_PerthAirport']=X['Location_PerthAirport'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Portland']=X['Location_Portland'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Richmond']=X['Location_Richmond'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Sale']=X['Location_Sale'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_SalmonGums']=X['Location_SalmonGums'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Sydney']=X['Location_Sydney'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_SydneyAirport']=X['Location_SydneyAirport'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Townsville']=X['Location_Townsville'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Tuggeranong']=X['Location_Tuggeranong'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Uluru']=X['Location_Uluru'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_WaggaWagga']=X['Location_WaggaWagga'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Walpole']=X['Location_Walpole'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Watsonia']=X['Location_Watsonia'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Williamtown']=X['Location_Williamtown'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Witchcliffe']=X['Location_Witchcliffe'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Wollongong']=X['Location_Wollongong'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X['Location_Woomera']=X['Location_Woomera'].apply(lambda x: 1.0 if x==1.0 else 0.0)
# X.drop(['Location'],axis=1,inplace=True)
                 

# X.info()

# # data cleaning for Y
# Y = Y.apply(lambda x: 1.0 if x == 'Yes' else 0.0)
# Y = Y.apply(lambda x: Y.mode() if np.isnan(x) else x)

# Y.info()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Read the CSV file
data = pd.read_csv('weatherAUS.csv')

# Separate target variable 'RainTomorrow' (Y) and features (X)
Y, X = data['RainTomorrow'], data.drop(['RainTomorrow'], axis=1)

# Data cleaning for X
# Drop irrelevant columns
X.drop(['Evaporation', 'Sunshine'], axis=1, inplace=True)

# Handling missing values with SimpleImputer (using median for numerical columns and mode for categorical columns)
imputer = SimpleImputer(strategy='median')
X[['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
   'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']] = imputer.fit_transform(X[
    ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
     'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']])

# Handling missing values in 'RainToday' with the mode
X['RainToday'].fillna(X['RainToday'].mode()[0], inplace=True)

# Handling missing values in wind direction columns with the mode
wind_direction_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
for column in wind_direction_columns:
    X[column].fillna(X[column].mode()[0], inplace=True)

# Encoding binary categorical column 'RainToday'
le = LabelEncoder()
X['RainToday'] = le.fit_transform(X['RainToday'])

# Encoding wind direction columns with OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse=False)
X = pd.concat([X.drop(wind_direction_columns, axis=1), pd.DataFrame(ohe.fit_transform(X[wind_direction_columns]),
                                                                    columns=ohe.get_feature_names(
                                                                        ['WindGustDir', 'WindDir9am', 'WindDir3pm']))],
              axis=1)

# Convert 'Date' column to datetime and drop rows with missing 'Date' values
X['Date'] = pd.to_datetime(X['Date'])
X.dropna(subset=['Date'], inplace=True)

# One-hot encoding for 'Location' column
X = pd.get_dummies(X, columns=['Location'])

# Data cleaning for Y
# Encoding target variable 'RainTomorrow' (Yes: 1, No: 0)
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now you have the cleaned and preprocessed X_train, Y_train, X_test, Y_test ready for building your machine learning model.
