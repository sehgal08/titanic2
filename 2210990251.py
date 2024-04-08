#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing= fetch_california_housing()


# In[2]:


housing


# In[3]:


housing.keys()


# In[5]:


x= pd.DataFrame(housing.data , columns =housing.feature_names)


# In[6]:


y= pd.DataFrame(housing.target, columns=housing.target_names)


# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def hypothesis(X, theta):
    return 1/(1 + np.exp(-(np.dot(X, theta))))

def cost(X, Y, theta):
    y_p = hypothesis(X, theta)
    loss = -1*(np.mean(Y*np.log(y_p + 1e-10) + (1- Y)*np.log(1 - y_p + 1e-10)))
    return loss

def gradient(X, Y, theta):
    y_p = hypothesis(X, theta)
    grad = np.dot(X.T , (Y - y_p))
    return grad/X.shape[0]

def gradient_descent(X, Y, learning_rate=0.3, epochs=100):
    m, n = X.shape
    theta = np.zeros((n , 1))
    cost_epoch = []

    for i in range(epochs):
        print('The algo is on epoch no : ',i, end='\r')
        loss = cost(X, Y, theta)
        grad = gradient(X, Y, theta)
        cost_epoch.append(loss)
        theta = theta + learning_rate * grad

    print(grad.shape, theta.shape)
    return theta, cost_epoch

def predict(X, theta):
    y_p = hypothesis(X, theta)
    y_pred = (y_p >= 0.5).astype(int)
    return y_pred

df = pd.read_csv("titanic_dataset.csv")
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna("S")
df["Male"] = pd.get_dummies(df['Sex'])["male"]
df["Male"] = df["Male"].map({True:0, False:1})
df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q":2})
df = df.drop(columns="Sex")
df['Age'] = df["Age"].astype('int64')
df['Fare'] = df["Fare"].astype('int64')
df = df.drop(columns=["Name", "Cabin", "Ticket", "PassengerId"])

X = df.drop(columns="Survived")
Y = df["Survived"]
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.2)
theta, cost = gradient_descent(X1, Y1, learning_rate=0.003, epochs=1000)
preds = predict(X2, theta)
accuracy_score(preds, Y2)
