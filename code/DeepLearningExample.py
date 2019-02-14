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
dataTrain=pd.read_csv('/home/pedromi/Descargas/MNIST/train.csv',header=0)
dataTest=pd.read_csv('/home/pedromi/Descargas/MNIST/test.csv',header=0)

train_labels=dataTrain['label']
train_images=dataTrain.iloc[:,1:]

network=models.Sequential()
network.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))

network.add(Flatten())
network.add(layers.Dense(500,activation='relu'))
network.add(layers.Dense(128))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images=train_images.values
train_images=train_images.reshape((42000,28,28,1))
train_images=train_images.astype('float32')/255

test_images=dataTest
test_images=test_images.values
test_images=test_images.reshape((28000,28,28,1))
test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)
#test_labels=to_categorical(test_labels)


network.fit(train_images,train_labels,epochs=15,batch_size=128)

pred=network.predict_classes(test_images)
#test_loss,test_acc=network.evaluate(test_images,test_labels)
print(pred)

df = pd.DataFrame(pred)
df.index+=1
df.to_csv('/home/pedromi/Descargas/MNIST/results.csv', index_label='ImageId',header=['Label'])