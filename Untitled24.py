#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


# Replace 'your_dataset.csv' with the actual filename of your dataset
data = pd.read_csv('C://Users//92300/Desktop//AAPL.csv')

# Assuming you have already processed and prepared the data, extract features and target variable
X = data.drop(['Date', 'Close'], axis=1)
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


data.head(10)


# In[4]:


# Create the XGBoost regressor
model = xgb.XGBRegressor()

# Train the model on the training data
model.fit(X_train, y_train)


# In[5]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE) to evaluate the model's performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[12]:


# Assuming you have a new data point 'new_data' for prediction
new_data = pd.DataFrame({'Open': [0.122210], 'High':[0.2], 'Low':[0.1], 'Adj Close':[0.087],'Volume': [175884800]})
predicted_price = model.predict(new_data)
print(f"Predicted stock price: {predicted_price[0]:.2f}")


# In[ ]:




