
#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#data processing
data=pd.read_csv('Social_Network_Ads.csv')
x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values


#feature scalling
from sklearn.preprocessing import StandardScaler 
standardscaler=StandardScaler()
x=standardscaler.fit_transform(x)


#traing resting  splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.25)


#kn neaighbours classfication
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
classifier.fit(x_train,y_train)

#prediction
y_pred=classifier.predict(x_test)



#comparing the result
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)