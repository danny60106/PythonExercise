#KNN
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

#下載資料
iris = datasets.load_iris()
x=iris.data
y=iris.target

#區分訓練集和測試集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#建模
#n_neighbor:K
#weights:'uniform'最好使用奇數K值 /'distance'加權 /其他
#algorithm:'auto'/'brute'/'kd_tree'/'ball_tree'
#p:1 曼哈頓距離 / p:2歐基里德距離 / 其他: 明氏距離
clf = KNeighborsClassifier(n_neighbors=3,p=2,weights='distance')
clf.fit(x_train,y_train)

#預測
clf.predict(x_test)

#準確程度評估
clf.score(x_test,y_test)
clf.score(x_train,y_train)

#尋找合適的K值
len(x_train)

accuracy = []

for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    
k_range = range(1,100)
plt.plot(k_range, accuracy)
plt.show()
