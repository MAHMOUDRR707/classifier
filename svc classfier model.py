#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#data processing
data=pd.read_csv('Social_Network_Ads.csv')
x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values



#feature scalling
from  sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


#trainging splitting  testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)


from sklearn.svm import SVC
scv=SVC(kernel='rbf',random_state=0)
scv.fit(x_train,y_train)

#predict the result
y_pred=scv.predict(x_test)


#comparing the result
from sklearn.metrics import confusion_matrix
sm=confusion_matrix(y_pred,y_test)