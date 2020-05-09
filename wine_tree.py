# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:27:30 2020

@author: sumanth
"""

import numpy as np
import pandas as pd
import os
#wine=pd.read_csv("H:\\data science\\wine.csv")
os.chdir("H:\\data science")

wine=pd.read_csv("wine.csv")
wine.shape
wine.dtypes
wine.describe()
list(wine)


x=wine.drop("Class",axis=1)
list(x)
x.shape

y=wine['Class']




###########################################################################
########################## Resampling method ##############################


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=25,stratify=y)
train_x.shape
train_y.shape
test_x.shape
test_y.shape

###########################################################################
############################# LOGISTIC REGRESSION #########################

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression().fit(train_x,train_y)
l=logr.score(test_x,test_y).round(3)
pred=logr.predict(test_x)
print("Logistic Regression accuracy value: ",l)

logte=[]
logtr=[]

for a in range(1,16):
    l=LogisticRegression(random_state=a).fit(train_x,train_y)
    logte.append(l.score(test_x,test_y).round(3))
    logtr.append(l.score(train_x,train_y).round(3))
print("Logistic Regression of sampling accuracy values: ",logte)


from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm=confusion_matrix(pred,test_y)
print("Accuracy :",metrics.accuracy_score(pred,test_y).round(3))


###########################################################################
########################### GINI ##########################################


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion="gini").fit(train_x,train_y)
#g=dtree.score(test_x,test_y).round(3)
print(f'tree which has {dtree.tree_.node_count} nodes and max depth {dtree.tree_.max_depth}')
predtree=dtree.predict(test_x)

cmtree=confusion_matrix(predtree,test_y)
print("Accuracy :",metrics.accuracy_score(predtree,test_y).round(3))


from sklearn.ensemble import RandomForestClassifier
rfG=RandomForestClassifier(n_estimators=500,criterion="gini").fit(train_x,train_y)
acc1=rfG.score(test_x,test_y).round(3)


print("In Decision Tree Classification of Random forest: gini accuracy value: ",acc1)
## cross validation
etr=[]
ete=[]

for l in range(1,16):
    w=RandomForestClassifier(n_estimators=l,criterion="gini").fit(train_x,train_y)
    etr.append(w.score(train_x,train_y).round(3))
    ete.append(w.score(test_x,test_y).round(3))
print("In Decision Tree Classification of Random forest: gini sampling accuracy values: ",ete)





from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(base_estimator=dtree,max_samples=0.5,n_estimators=500,random_state=10)
bag.fit(train_x,train_y)
bag.score(test_x,test_y).round(3)

btr=[]
bte=[]
for a in range(1,11):
    bagg=BaggingClassifier(base_estimator=dtree,max_samples=0.5,n_estimators=500,random_state=a)
    bagg.fit(train_x,train_y)
    btr.append(bagg.score(train_x,train_y).round(3))
    bte.append(bagg.score(test_x,test_y).round(3))

print("BaggingClassifier accuracy values in gini :",bte)




#predrfG=rfG.predict(test_x)
#rftreeG=confusion_matrix(predrfG,test_y)
#print("Accuracy :",metrics.accuracy_score(rftreeG,test_y).round(3))


###########################################################################
########################## ENTROPY ########################################


dtreeEnt=DecisionTreeClassifier(criterion="entropy").fit(train_x,train_y)
print(f'tree which has {dtreeEnt.tree_.node_count} nodes and max depth {dtreeEnt.tree_.max_depth}')
predtreeEnt=dtree.predict(test_x)

cmtreeEnt=confusion_matrix(predtreeEnt,test_y)
print("Accuracy :",metrics.accuracy_score(predtreeEnt,test_y).round(3))

from sklearn.ensemble import RandomForestClassifier
rfEnt=RandomForestClassifier(n_estimators=500,max_features=0.5).fit(train_x,train_y)
acc=rfEnt.score(test_x,test_y).round(3)
print("In Decision Tree Classification of Random forest: entropy accuracy value: ",acc)
## cross validation
gtr=[]
gte=[]

for l in range(1,16):
    q=RandomForestClassifier(n_estimators=500,criterion="entropy",random_state=l).fit(train_x,train_y)
    gtr.append(q.score(train_x,train_y).round(3))
    gte.append(q.score(test_x,test_y).round(3))

print("In Decision Tree Classification of Random forest: entropy sampling accuracy values: ",gte)





bag=BaggingClassifier(base_estimator=dtreeEnt,max_samples=0.5,n_estimators=500,random_state=10)
bag.fit(train_x,train_y)
bag.score(test_x,test_y).round(3)

btr1=[]
bte1=[]
for aa in range(1,11):
    bagg=BaggingClassifier(base_estimator=dtreeEnt,max_samples=0.5,n_estimators=500,random_state=aa)
    bagg.fit(train_x,train_y)
    btr1.append(bagg.score(train_x,train_y).round(3))
    bte1.append(bagg.score(test_x,test_y).round(3))

print("BaggingClassifier accuracy values in Entropy :",bte1)

#predrfEnt=rfEnt.predict(test_x)
#rftreeEnt=confusion_matrix(predrfEnt,test_y)
#print("Accuracy :",metrics.accuracy_score(rftreeEnt,test_y).round(3))


















































































 