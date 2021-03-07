#Google colab link:https://colab.research.google.com/drive/1Knd-ThrWqrqF5-uWmMp9sK0xiAwWQnbE?usp=sharing
#Neural Net fot Multiclass Classification
#switched from sigmoid to softmax activation function
#One hot incoding
#instead of binary cross entropy, now using categorical cross entropy to calculate the accuracy of the model
#then use gradient decent to minimise error obtained readjust the weights with back propagation

import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets
from keras.models import Sequential #define model
from keras.layers import Dense #connect proceeding layers to subsequent layers
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


#Multiclass dataset
n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]] #centers of datapoints in DS,
X, y = datasets.make_blobs(n_samples = n_pts, random_state=123, centers=centers, cluster_std = 0.4)

plt.scatter(X[y==0, 0], X[y==0, 1]) #boolean function, fetch all the lables 'y' that are at index '0'. and so on...
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])

#Hotencoding process
print(y)
y_cat = to_categorical(y, 5) # 'y' is labels of DS points, interger value of the no. of Data classes
print(y_cat)


#NN Model to classify a multi class Dataset
#Needs two nodes of input (X & Y coordinate) and three outputs of hotencode (001,010,100)
model = Sequential()
model.add(Dense(units=5, input_shape=(2,), activation='softmax'))
model.compile(Adam(0.1), loss = 'categorical_crossentropy', metrics=['accuracy'])


#model Training
model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs= 100)


# with the trained model, correlated each coordinate with prediction prob 'p', plotting contours, representing distinc prob levels. TF ploting classifaction boundary i.e decision Boundary with DS
def plot_decision_boundary(X, y_cat, model):
  x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0])+ 1, 50)
  y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1])+ 1, 50)
  xx, yy = np.meshgrid(x_span, y_span)
  xx_, yy_ = xx.ravel(), yy.ravel()
  print(xx_)
  print(yy_)
  grid = np.c_[xx_, yy_]
  pred_func = model.predict_classes(grid)
  z = pred_func.reshape(xx.shape)
  plt.contourf(xx, yy, z)


plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1]) #boolean function, fetch all the lables 'y' that are at index '0'. and so on...
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])
#purple = lable 0
#blue = lable 1
#yellow = lable 2



plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1]) #boolean function, fetch all the lables 'y' that are at index '0'. and so on...
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])
#test point input
x = 0.5
y = -1
point = np.array([[x, y]])
#obtain a predciton value
prediction = model.predict_classes(point)
#plot point on
plt.plot([x], [y], marker='o', markersize=10, color="r")
print("prediction is ", prediction)
