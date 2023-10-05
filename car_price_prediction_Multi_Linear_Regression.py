
# Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Read file

df = pd.read_csv('CarPrice_Assignment.csv')
data_set = df.copy()
data_set.dtypes

# Data cleaning
data_set.isnull().sum(axis=0)
data_set = data_set.drop(['car_ID'], axis=1)

# Identify the data types of features

numerical_features = data_set.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data_set.select_dtypes (include=['object']).columns

# Group numerical & categorical features

numerical_data = data_set[numerical_features]
categorical_data = data_set[categorical_features]

numarical_summary = numerical_data.describe()
categorical_cunts = categorical_data.nunique()


# create afigure and axis objects for subplots

fig, axs = plt.subplots(3,5, figsize=(19,10))
axs = axs.flatten()

for i , feature in enumerate(numerical_data):
    sns.scatterplot(data=data_set, x=feature, y='price', ax=axs[i])
    
# from the scatterplot we get features like wheelbase, carlength, carwidth, curbweight,\
# enginesize, boneration, horsepower, citympg, highwaympg are linearly relationship with \
# price

# wheelbase, carlength, boreratio are seemed to be having outliers 

# and citympg and highwaympg having negative relationship with price .


# craeting box plot to confirm outliers in wheelbase , carlength, boneratio 

fig, axs = plt.subplots(1, 3, figsize=(10,2))
axs = axs.flatten()

for i , feature in enumerate(['wheelbase', 'carlength', 'boreratio']):
    sns.boxplot(data=data_set, x=feature, ax=axs[i])
    
    

# create the heatmap using seaborn

plt.figure(figsize=(15,10))
cor = numerical_data.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.show()


# Multi-collinearity features found :-

# highwaympg – carlength
# enginesize – curbweight
# curbweight – carlength
# highwaympg – horsepower
# curbweight – carwidth
# enginesize – carwidth
# carlength – highwaympg
# curbweight – highwaympg
# curbweight – citympg
# carlength – wheelbase
# enginesize – horsepower
# citympg – horsepower
# highwaympg - citympg


# After checking multi-collinearity, decided to go with the below features
# 'carlength', 'enginesize', 'horsepower','boreratio'

# will drop features 
# 'symboling', 'wheelbase', 'carwidth', 'carheight', 'curbweight', 'stroke', \
# 'compressionratio', 'peakrpm', 'citympg', 'highwaympg' 



numerical_data = numerical_data.drop(['symboling', 'wheelbase', 'carwidth', 'carheight', 'curbweight', 'stroke','compressionratio', 'peakrpm', 'citympg', 'highwaympg' ], axis=1 )

                                      
    
# in categorical_data CarName is irrelevant for the price prediction , so drop it .

categorical_data = categorical_data.drop(['CarName'], axis=1)    

# Unique values of each features.
for feature in categorical_data.columns:
    print(feature)
    print(categorical_data[feature].unique())
    

# Apply one-hot encoding to the remaining categorical features
encoded_categorical_data = pd.get_dummies(categorical_data)
encoded_categorical_data.head()    

# feature scalling
target = numerical_data['price'] 
continous_features =['carlength', 'enginesize', 'horsepower', 'boreratio']   
scaler = StandardScaler()    
scaled_feature = scaler.fit_transform(numerical_data[continous_features])    

# Create a DataFrame with the scaled features and feature names
scaled_df =pd.DataFrame(scaled_feature,columns=continous_features)    

final_df = pd.concat([encoded_categorical_data ,scaled_df,target], axis=1)


# Split training & testing data
X = final_df.iloc[ : , :-1]
y = final_df.iloc[ : , -1]

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


# Running Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train) 
y_predict = lr.predict(X_test) 

# Mean Squard Error
lr_mse = mean_squared_error(y_test, y_predict)  
# mse = 9004377.96651616

# Get the R-Squared 
lr_score = lr.score(X_test, y_test)    
# R squard = 0.8517054733102433

# Root mean squard error
lr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
# rmse = 3000.729572373385
















