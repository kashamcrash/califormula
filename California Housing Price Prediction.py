#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Credentials - Karan Sharma / kasham1991@gmail.comAnalysis Tasks to be performed:

# 1. Build a model of housing prices to predict median house values in California using the provided dataset.
# 2. Train the model to learn from the data to predict the median housing price in any district, given all the other metrics.
# 3. Predict housing prices based on median_income and plot the regression chart for it.


# In[30]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[31]:


# Importing the dataset
housing = pd.read_excel('C:\\Datasets\\Housing.xlsx')
housing


# In[32]:


housing.info()
# housing.size
# housing.shape

# Exploratory Data Analysis
# 20640 corresponds to the number of districts, 10 responds to the characteristics prevailing in each district
# One categorical variable - ocean_proximity
# median_house_value is the median sales of each district


# In[33]:


# Looking at the basic statistics
housing.describe()
housing.describe().T


# In[34]:


# Checking for null values
# total bedrooms has 207 NaN values
housing.isnull()
housing.isnull().sum()


# In[35]:


# Dropping null values
housing.dropna(inplace = True)


# In[36]:


housing.isnull().sum()


# In[37]:


housing.columns


# In[38]:


# Encode categorical data - Convert categorical to numerical data
# The feature column ocean_proximity is in categorical format
# The are multiple ways to convert categorical data - label encoding, binary encone, dummies, etc
# Pandas supports many more methods for the same

# Label encoding involves converting each value from the column into a number
# Ocean_proximity -'Near Bay' | '<1H Ocean' | 'Inland' | 'Near Ocean' | 'Island'
# Assigning a specific number 
# <1H Ocean = 0, Inland = 1, Island = 2, Near Bay = 3, Near Ocean = 4

from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()


# In[39]:


# Converting the column from object to category
housing['ocean_proximity'] = housing['ocean_proximity'].astype('category')
housing.dtypes


# In[40]:


# Assigning the encoded variable to a new column using the cat.codes accesor
housing['ocean_proximity_value'] = housing['ocean_proximity'].cat.codes
housing.head()


# In[41]:


# Since we are using regression analysis, it is better to standardize the data
# Standardization involves shifting the distribution of each data point to a mean of 0 and an SD of 1
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x =  pd.DataFrame(sc_x.fit_transform(housing.drop(["ocean_proximity", "median_house_value"], axis = 1),),
        columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity_value'])
x.head()
# x.shape


# In[42]:


# Standardizing the target variable - median_house_value
y = pd.DataFrame(sc_x.fit_transform(housing.drop(['ocean_proximity','longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity_value' ], axis = 1),),
                 columns=['median_house_value'])
y.head()
# y.shape


# In[43]:


# splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 1,)


# In[44]:


# Linear Regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train, y_train)


# In[45]:


# Printing the coefficients and intercept
print(model1.intercept_)
print(model1.coef_)


# In[46]:


# Predicting on the test data
y_p = model1.predict(x_test)
print(y_p)
print(y_test)


# In[47]:


# Calculating the Root Mean Squared Error - RMSE
# Test and train scores
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_p)))
print(np.sqrt(metrics.mean_squared_error(y_train, model1.predict(x_train))))


# In[48]:


# Univariate regression
x1 = x[['median_income']]
y1 = y[['median_house_value']]


# In[49]:


# Splitting the data
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = .20, random_state = 1,)


# In[50]:


# Linear Regression
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(x1_train, y1_train)


# In[51]:


# Printing the coefficients and intercept
print(model2.intercept_)
print(model2.coef_)


# In[52]:


# Predicting on the test data
y_p1 = model2.predict(x1_test)
print(y_p1)
print(y1_test)


# In[53]:


# Calculating the Root Mean Squared Error - RMSE
# Test and train scores
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y1_test, y_p1)))
print(np.sqrt(metrics.mean_squared_error(y1_train, model2.predict(x1_train))))


# In[54]:


# Lets plot the fitted model2 with the revised dataset
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(y_p1[0:25], 'y')
plt.plot(y1[0:25], 'b')
plt.plot(x1[0:25], 'g')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Model Prediction Chart')


# In[ ]:


# Thank You :) 

