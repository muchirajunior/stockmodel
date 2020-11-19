#import the libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
import random

#import csv data into a pandas dataframe
df=pd.read_csv('data.csv')

#declare the processing arrays
dates=[]
prices=[]
yprices=[]

#partition the dataframe by removing the last 5 rows
df=df.head(len(df)-5)

#get dates  and prices in open columns that will be used to 
# make predictions from the data frame
df_dates=df['Date']
df_prices=df['open']
df_close=df['Close/Last']

#cleaning the data for processing and feed them into the processing arrays
#process dates
for item in df_dates:
    dates.append(item[3:5])

#process input prices
for item in df_prices:
    prices.append(item[2:])

# process output prices
for item in df_close:
    yprices.append(item[2:])


#make a multidimensional array named data that will incorporate all data that was processed 
data=[]
for item in range(len(dates)):
    data.append([dates[item],prices[item],yprices[item]])

#use the multidimensional array to make a new data frame where yprices are the output prices
train=pd.DataFrame(data,columns=['dates','prices','yprice'])

#make an input data Xtrain for the model by dropping the output prices column that is yprices
Xtrain=train.drop(columns='yprice')

#make the output data  column ytrain for the model
ytrain=train['yprice']


#make a decision tree classifier model 
model=DecisionTreeClassifier()
model.fit(Xtrain,ytrain)

#make 3 support vector regression models, linear poly and rbf 
svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
    
#train the models
svr_lin.fit(Xtrain,ytrain)
svr_poly.fit(Xtrain,ytrain)
svr_rbf.fit(Xtrain,ytrain)
    
#create a linear Regression model
lin_reg=LinearRegression()
#train the linear regression model
lin_reg.fit(Xtrain,ytrain)

# make ploting data arrays to pick 10 radom value from the dataset and plot
xplot=random.sample(dates,10)
yplot=random.sample(yprices,10)

class Model:
    def __init__():
        myname='MUCHIRA JUNIOR'
        # the constructer is empty
    
    #a fuction that line and scatter of the data
    def linescatterplot():
        plt.plot(xplot,yplot,color='blue',label=' Regression')
        plt.scatter(xplot,yplot,color='black',label=' Reg scatter')
        plt.title('LINE & SCATTER PLOT OF THE DATA')
        plt.xlabel('DAYS')
        plt.ylabel('PRICES')
        plt.savefig('lineplot.png',bbox_inches='tight')
    
    #plotting a bar graph of the data dates against output prices
    def bargraphplot():
        plt.title('BAR GRAPH PLOT OF THE DATA')
        plt.xlabel('DAYS')
        plt.ylabel('PRICES')
        plt.bar(xplot,yplot)
        plt.savefig('barplot.png',bbox_inches='tight')

    #common model prediction
    def normal_model(x,y):
        prediction=model.predict([[0,0],[x,y]])
        return prediction[1]

    #support vector machine linear model prediction
    def svr_linear_model(x,y):
        prediction=svr_lin.predict([[0,0],[x,y]])
        return prediction[1]

    #support vector machine ploy model prediction
    def svr_poly_model(x,y):
        prediction=svr_poly.predict([[0,0],[x,y]])
        return prediction[1]

    #support vector machine rbf model prediction
    def svr_rbf_model(x,y):
        prediction=svr_rbf.predict([[0,0],[x,y]])
        return prediction[1]

    #common linear regression model prediction
    def linear_model(x,y):
        prediction=lin_reg.predict([[0,0],[x,y]])
        return prediction[1]

