#!/usr/bin/env python
# coding: utf-8

# # CSE474/574 - Programming Assignment 1
# 
# For grading, we will execute the submitted notebook as follows:
# 
# ```shell
# jupyter nbconvert --to python PA1Script.ipynb
# python PA1Script.py
# ```

# In[1]:


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle


# ## Part 1 - Linear Regression

# ### Problem 1 - Linear Regression with Direct Minimization

# In[2]:


print('PROBLEM 1')
print('Linear Regression with Direct Minimization')


# In[3]:


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    X_trans = np.transpose(X)
    x = np.dot(X_trans,X)
    y = np.dot(X_trans,y)
    inverse = np.linalg.inv(x)
    w = np.dot(inverse,y)
    print(w.shape)
    return w


# In[4]:


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    w_trans = np.transpose(w)
    X_trans = np.transpose(Xtest)
    Y_trans = np.transpose(ytest)
    
    N = 1/Xtest.shape[0]
    a = np.dot(w_trans, X_trans)
    b = np.subtract(Y_trans, a)
    c = np.power(b,2)
    d = np.dot(c, N)
    sigma = np.sum(d)
    rmse = np.sqrt(sigma)
    
    #rmse = 0
    return rmse


# In[5]:


X_train,y_train,X_test,y_test = pickle.load(open('diabetes.pickle','rb'),encoding='latin1')   
# add intercept
x1 = np.ones((len(X_train),1))
x2 = np.ones((len(X_test),1))

X_train_i = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis=1)
X_test_i = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)

w = learnOLERegression(X_train,y_train)
w_i = learnOLERegression(X_train_i,y_train)

rmse = testOLERegression(w,X_train,y_train)
rmse_i = testOLERegression(w_i,X_train_i,y_train)
print('RMSE without intercept on train data - %.2f'%rmse)
print('RMSE with intercept on train data - %.2f'%rmse_i)

rmse = testOLERegression(w,X_test,y_test)
rmse_i = testOLERegression(w_i,X_test_i,y_test)
print('RMSE without intercept on test data - %.2f'%rmse)
print('RMSE with intercept on test data - %.2f'%rmse_i)


# ### Problem 2 - Linear Regression with Gradient Descent

# In[6]:


print('PROBLEM 2')
print('Linear Regression with Gradient Descent')


# In[7]:


def regressionObjVal(w, X, y):

    # compute squared error (scalar) with respect
    # to w (vector) for the given data X and y      
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    a = np.dot(X,w)
    error = 0.5*(np.dot(np.subtract(y, np.transpose(a)), np.subtract(y,a)))[0][0]
    #error = 0
    return error


# In[8]:


def regressionGradient(w, X, y):

    # compute gradient of squared error (scalar) with respect
    # to w (vector) for the given data X and y   
    
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # gradient = d length vector (not a d x 1 matrix)

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE 
    # X^T Xw - X^T y
    a = np.transpose(X)
    b = np.dot(X,w)
    c = np.dot(a,b)
    d = np.dot(a,y)
    error_grad = np.subtract(c,d)
    
    error_grad = [i[0] for i in error_grad]
    
    error_grad = np.array(error_grad)
    
    #error_grad = np.zeros((X.shape[1],))
    return error_grad


# In[9]:


Xtrain,ytrain,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding='latin1')   
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
args = (Xtrain_i,ytrain)
opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))
soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args,method='CG', options=opts)
w = np.transpose(np.array(soln.x))
w = w[:,np.newaxis]
rmse = testOLERegression(w,Xtrain_i,ytrain)
print('Gradient Descent Linear Regression RMSE on train data - %.2f'%rmse)
rmse = testOLERegression(w,Xtest_i,ytest)
print('Gradient Descent Linear Regression RMSE on test data - %.2f'%rmse)


# ## Part 2 - Linear Classification

# ### Problem 3 - Perceptron using Gradient Descent

# In[10]:


print('PROBLEM 3')
print('Perceptron using Gradient Descent')


# In[11]:


def predictLinearModel(w,Xtest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # Output:
    # ypred = N x 1 vector of predictions

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    N = len(Xtest)
    ypred = np.zeros([Xtest.shape[0],1])
    for i in range(1,N):
        #a = np.dot(w.transpose(), Xtest[i])
        if np.dot(np.transpose(w), Xtest[i]) >= 0:
            ypred[i]=1
        else:
            ypred[i]=-1
    return ypred


# In[12]:


def evaluateLinearModel(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # acc = scalar values

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    acc = 0
    ypred = predictLinearModel(w,Xtest)
    N = len(Xtest)
    for i in range(1,N):
        if ypred[i] == ytest[i]:
            acc += 1
    #acc = 0
    return acc


# In[13]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

args = (Xtrain_i,ytrain)
opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))
soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args,method='CG', options=opts)
w = np.transpose(np.array(soln.x))
w = w[:,np.newaxis]
acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print('Perceptron Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('Perceptron Accuracy on test data - %.2f'%acc)


# ### Problem 4 - Logistic Regression Using Newton's Method

# In[14]:


print('PROBLEM 4')
print('Logistic Regression Using Newton Method')


# In[15]:


def logisticObjVal(w, X, y):

    # compute log-loss error (scalar) with respect
    # to w (vector) for the given data X and y                               
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar
    
    N = len(X)
    error = 0
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    
    for i in range(0, N):
        error += np.dot(1/N, np.log(1 + np.exp(np.dot(np.dot(-1*y[i], w.transpose()), X[i]))))
    
    return error


# In[16]:


def logisticGradient(w, X, y):

    # compute the gradient of the log-loss error (vector) with respect
    # to w (vector) for the given data X and y  
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = d length gradient vector (not a d x 1 matrix)
    
    N = len(Xtest)
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
        
    gradient = np.zeros((w.shape[0],))
    
    for i in range(0, N):
        gradient += np.dot(np.divide(y[i], 1 + np.exp(np.dot(np.dot(y[i], w.transpose()), X[i]))).reshape(-1,1), X[i].reshape(1,3)).reshape(-1)

    gradient = np.dot(-1/N, gradient)
    
    return gradient


# In[17]:


def logisticHessian(w, X, y):

    # compute the Hessian of the log-loss error (matrix) with respect
    # to w (vector) for the given data X and y                               
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # Hessian = d x d matrix
    
    N = len(X)
    
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
        
    hessian = np.eye(X.shape[1])

    for i in range(0, N):
        hessian += np.dot(np.dot(np.divide(np.exp(np.dot(np.dot(y[i], w.transpose()), X[i])),(1+np.exp(np.dot(np.dot(y[i], w.transpose()), X[i])))**2), X[i]), X[i].transpose())
        
    hessian = np.dot((1/N), hessian)

    return hessian


# In[18]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

args = (Xtrain_i,ytrain)
opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))
soln = minimize(logisticObjVal, w_init, jac=logisticGradient, hess=logisticHessian, args=args,method='Newton-CG', options=opts)
w = np.transpose(np.array(soln.x))
w = np.reshape(w,[len(w),1])
acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print('Logistic Regression Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('Logistic Regression Accuracy on test data - %.2f'%acc)


# ### Problem 5 - Support Vector Machines Using Gradient Descent

# In[19]:


print('PROBLEM 5')
print('Support Vector Machines Using Gradient Descent')


# In[20]:


def trainSGDSVM(X,y,T,eta=0.01):
    # learn a linear SVM by implementing the SGD algorithm
    #
    # Inputs:
    # X = N x d
    # y = N x 1
    # T = number of iterations
    # eta = learning rate
    # Output:
    # weight vector, w = d x 1
    
    # IMPLEMENT THIS METHOD
    from random import randint
    N = len(X)
    w = np.zeros([X.shape[1],1])
    w = w.reshape(-1)
    
    for i in range (0, T):
        j = randint(0, N-1)
            
        if np.dot(np.dot(y[j].reshape(-1,1), w.reshape(1,3)), X[j]) < 1:
            addition = np.dot(np.dot(eta, y[j].reshape(-1,1)), X[j].reshape(1,3))
            w = np.add(w, addition.reshape(-1))
    
    return w


# In[21]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

args = (Xtrain_i,ytrain)
w = trainSGDSVM(Xtrain_i,ytrain,200,0.01)
acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print('SVM Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('SVM Accuracy on test data - %.2f'%acc)


# ### Problem 6 - Plotting decision boundaries

# In[22]:


print('Problem 6')
print('Plotting decision boundaries')


# In[23]:


def plotBoundaries(w,X,y):
    # plotting boundaries

    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    x1 = np.linspace(mn[1],mx[1],100)
    x2 = np.linspace(mn[2],mx[2],100)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.ravel()
    xx[:,1] = xx2.ravel()
    xx_i = np.concatenate((np.ones((xx.shape[0],1)), xx), axis=1)
    ypred = predictLinearModel(w,xx_i)
    ax.contourf(x1,x2,ypred.reshape((x1.shape[0],x2.shape[0])),alpha=0.3,cmap='cool')
    ax.scatter(X[:,1],X[:,2],c=y.flatten())


# In[24]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

# Replace next three lines with code for learning w using the three methods
# w_perceptron = np.zeros((Xtrain_i.shape[1],1))
# w_logistic = np.zeros((Xtrain_i.shape[1],1))
# w_svm = np.zeros((Xtrain_i.shape[1],1))
###################################
soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args,method='CG', options=opts)
w_perceptron = w

soln = minimize(logisticObjVal, w_init, jac=logisticGradient, hess=logisticHessian, args=args,method='Newton-CG', options=opts)
w_logistic = w

w = trainSGDSVM(Xtrain_i,ytrain,200,0.01)
w_svm = w
####################################
fig = plt.figure(figsize=(20,6))

ax = plt.subplot(1,3,1)
plotBoundaries(w_perceptron,Xtrain_i,ytrain)
ax.set_title('Perceptron')

ax = plt.subplot(1,3,2)
plotBoundaries(w_logistic,Xtrain_i,ytrain)
ax.set_title('Logistic Regression')

ax = plt.subplot(1,3,3)
plotBoundaries(w_svm,Xtrain_i,ytrain)
ax.set_title('SVM')


# In[ ]:




