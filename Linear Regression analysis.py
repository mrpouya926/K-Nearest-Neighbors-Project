#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Linear Regression Project
# 
# Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!
# 
# Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you real credit card numbers or emails).

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[3]:


customers = pd.read_csv('Ecommerce Customers')


# In[ ]:





# In[4]:


customers.head()


# In[7]:


customers.describe()


# In[8]:


customers.info()


# In[279]:





# In[11]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent', data= customers)


# In[12]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent', data= customers)


# In[ ]:





# In[15]:


sns.jointplot(x='Time on App',y='Length of Membership', data= customers, kind='hex')


# In[16]:


sns.pairplot(data =customers)


# In[ ]:


#Length of Membership look to be the most correlated feature with Yearly Amount spent.


# In[285]:





# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[22]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data= customers);


# ## Training and Testing Data

# In[60]:


customers.columns


# In[71]:


X =customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[72]:


y = customers["Yearly Amount Spent"]


# In[ ]:





# In[73]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:





# ## Training the Model
# 
# Now its time to train our model on our training data!

# In[75]:


from sklearn.linear_model import LinearRegression


# In[76]:


lm = LinearRegression()


# In[77]:


lm.fit(X_train,y_train)


# In[293]:





# In[82]:


lm.coef_


# In[ ]:





# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 

# In[91]:


predictions = lm.predict(X_test)


# In[ ]:





# ** Create a scatterplot of the real test values versus the predicted values. **

# In[93]:


sns.scatterplot(x=y_test, y=predictions, data = customers)


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[94]:


from sklearn import metrics


# In[95]:


metrics.mean_absolute_error(y_test, predictions)


# In[96]:


metrics.mean_squared_error(y_test, predictions)


# In[98]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[101]:


sns.distplot(y_test-predictions, bins=40)


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# There is two way to think about this issue. if you look at your coefficients and the data in our model we could develop the website to catch up with the mobile app. So, one way of thinking about this is the website needs the most work. on the other side, we can think of it another way which is you should develop the app more since it is working already much better and this sort of answer depends on a number of factors going on at this particular company or wherever this data is actually is coming from and we should explore the length of the relationship between of membership and the app or the web site for coming to a conclusion.

# In[105]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeffecient'])


# In[106]:


cdf


# In[ ]:




