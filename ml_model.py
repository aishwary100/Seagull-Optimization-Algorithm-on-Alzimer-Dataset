# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:51:00 2021

@author: yash
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train_SOA=np.load("numpy_files/X_train_SOA.npy")
y_train_SOA=np.load("numpy_files/y_train_SOA.npy")
X_test_SOA=np.load("numpy_files/X_test_SOA.npy")
y_test_SOA=np.load("numpy_files/y_test_SOA.npy")

print('X_test_SOA.shape',X_test_SOA.shape)
print('X_train_SOA.shape',X_train_SOA.shape)
print('Y_test_SOA.shape',y_train_SOA.shape)
print('y_test_SOA.shape',y_test_SOA.shape)

########################### [Classifier] #############################################

start_time = time.time()
classifier = SVC(kernel = 'linear')
classifier.fit(X_train_SOA, y_train_SOA)
print("\n--- %s seconds ---" % (time.time() - start_time))
print("[Classifier] :", classifier.score(X_test_SOA,y_test_SOA))
#y_pred = classifier.predict(X_test_WOA)
#print(confusion_matrix(y_test_WOA, y_pred))
#print(classification_report(y_test_WOA,y_pred))

########################### [K-neighbour Classifier] #############################

start_time = time.time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_SOA, y_train_SOA)
print("\n--- %s seconds ---" % (time.time() - start_time))
print("[K-neighnbour Classifier] :",neigh.score(X_test_SOA,y_test_SOA))

########################### [Decision-tree-Classifier] #############################

clf = tree.DecisionTreeClassifier(random_state=0)
start_time = time.time()
clf.fit(X_train_SOA,y_train_SOA)
print("\n--- %s seconds ---" % (time.time() - start_time))
print("[Decision-tree-Classifier] :",clf.score(X_test_SOA,y_test_SOA))


########################### [Random-forest Classifier] #############################

clf = RandomForestClassifier(max_depth=4, random_state=0)
start_time=time.time()
clf.fit(X_train_SOA,y_train_SOA)
print("\n--- %s seconds ---" % (time.time() - start_time))
print("[Random-forest Classifier] :",clf.score(X_test_SOA,y_test_SOA))

