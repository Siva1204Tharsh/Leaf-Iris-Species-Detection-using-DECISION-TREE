##Import Basic Libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

##Load Dataset
dataset=load_iris()
# print(dataset.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

##Summarize dataset
# print(dataset.data) 
# print(dataset)

##Segregating dataset into X and Y
X=pd.DataFrame(dataset.data,columns=dataset.feature_names)
# print(X)
Y=dataset.target
# print(Y)

##Splitting Dataset into train & test 
from sklearn.model_selection import train_test_split

X_train,X_test ,y_train ,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# print(X_train.shape) #(120, 4)
# print(X_test.shape) #(30, 4)

# ##Finding best max_depth value
# accuracy=[]
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# for i in range(1,10):
#     model=DecisionTreeClassifier(max_depth=i,random_state=0)
#     model.fit(X_train,y_train)
#     pred=model.predict(X_test)
#     score=accuracy_score(y_test,pred)
#     accuracy.append(score)
# plt.figure(figsize=(12,6))
# plt.plot(range(1,10),accuracy,color='red',marker='o',markerfacecolor='blue',markersize=10)
# plt.title('Finding best Max_depth')
# plt.xlabel('pred')
# plt.ylabel('score')
# plt.show() 

#i=4
##Model Training 

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=0)
model.fit(X_train,y_train)

##prediction of model 
y_pred=model.predict(X_test)
print(y_pred)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

##model Accuracy score
from sklearn.metrics import accuracy_score
print("Accuracy of the model:{0}%".format(accuracy_score(y_test,y_pred)*100))