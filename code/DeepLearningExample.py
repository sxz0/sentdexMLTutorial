#Build a neural network to clasify handwritten digits
#
#First example->Deep Learning with Python
from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.cm import cmap_d
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
dataTrain=pd.read_csv('train_data_chema.csv',header=0,index_col=0)
dataTest=pd.read_csv('evaluation_data_chema.csv',header=0,index_col=0)

train_labels=dataTrain['threatLevel']
train_data=dataTrain.drop("threatLevel",1)
test_labels=dataTest['threatLevel']
test_data=dataTest.drop("threatLevel",1)

network=models.Sequential()
#network.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#network.add(MaxPooling2D(pool_size=(2, 2)))
#network.add(Dropout(0.25))
print(train_data.shape)
print(test_data.shape)
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
#train_data=train_data.values.reshape(7,1,1)
#test_data=test_data.values.reshape(7,1,1)
#network.add(Flatten())
network.add(layers.Dense(50,activation='relu',input_shape=[7]))
network.add(layers.Dense(20))
network.add(layers.Dense(10))
network.add(layers.Dense(8))
network.add(layers.Dense(4,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


network.fit(train_data,train_labels,epochs=15,batch_size=128)

pred=network.predict_classes(test_data)
test_loss,test_acc=network.evaluate(test_data,test_labels)

print(test_loss,"  ",test_acc)
