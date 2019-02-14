from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.cm import cmap_d
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

dataTrain=pd.read_csv('/home/pedromi/Descargas/iniciacionSpainML-Credit/train.csv',header=0)
dataTest=pd.read_csv('/home/pedromi/Descargas/iniciacionSpainML-Credit/test.csv',header=0)

train_labels=dataTrain['default.payment.next.month']
train_data=dataTrain.iloc[:,1:24]
test_data=dataTest.iloc[:,1:]
all_data=pd.concat([train_data,test_data])

#PREPROCESAMIENTO
normalized_df=(all_data-all_data.mean())/all_data.std()
all_data=normalized_df
all_data = pd.concat([all_data,pd.get_dummies(all_data['SEX'], prefix='SEX')],axis=1)
all_data.drop(['SEX'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['EDUCATION'], prefix='EDUCATION')],axis=1)
all_data.drop(['EDUCATION'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['MARRIAGE'], prefix='MARRIAGE')],axis=1)
all_data.drop(['MARRIAGE'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_0'], prefix='PAY_0')],axis=1)
all_data.drop(['PAY_0'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_2'], prefix='PAY_2')],axis=1)
all_data.drop(['PAY_2'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_3'], prefix='PAY_3')],axis=1)
all_data.drop(['PAY_3'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_4'], prefix='PAY_4')],axis=1)
all_data.drop(['PAY_4'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_5'], prefix='PAY_5')],axis=1)
all_data.drop(['PAY_5'],axis=1, inplace=True)
all_data = pd.concat([all_data,pd.get_dummies(all_data['PAY_6'], prefix='PAY_6')],axis=1)
all_data.drop(['PAY_6'],axis=1, inplace=True)

all_data=all_data[['PAY_0_1.7945339544019632','AGE','LIMIT_BAL','BILL_AMT1', 'PAY_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT3', 'PAY_AMT2', 'BILL_AMT5', 'BILL_AMT6', 'PAY_2_1.7823184657421385', 'PAY_AMT1', 'BILL_AMT2', 'BILL_AMT4']]
#pca = PCA(n_components=50)
#pca.fit(all_data)

train_data=all_data.iloc[:29000,:]
#train_data=pca.transform(train_data)

test_data=all_data.iloc[29000:,:]
#test_data=pca.transform(test_data)

X_train,X_cross,y_train,y_cross=train_test_split(train_data,train_labels,test_size=0.10,stratify=train_labels,random_state=1)

rf = RandomForestRegressor()
rf.fit(train_data, train_labels)
names = all_data.columns.values
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True))

#clasif=svm.SVC(gamma=0.1)
#clasif.fit(X_train,y_train)
#pred=clasif.predict(X_cross)
#accuracy=clasif.score(X_cross,y_cross)

#log= LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_data,train_labels)
#pred=log.predict(test_data)
#pred_proba_df = pd.DataFrame(log.predict_proba(test_data))
#pred = pred_proba_df.applymap(lambda x: 1 if x>0.3 else 0)
#pred=pred.iloc[:,1]
#print (f1_score(y_cross, pred, average='macro'))

#accuracy=log.score(X_cross,y_cross)


network=models.Sequential()
network.add(layers.Dense(500,activation='relu', input_shape=(15,)))
network.add(layers.Dense(50))
network.add(layers.Dense(2,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_labels=to_categorical(train_labels)
y_train=to_categorical(y_train)

network.fit(train_data,train_labels,epochs=50,batch_size=128)

pred=network.predict_proba(test_data)
pred_proba_df = pd.DataFrame(pred)
#for i in np.arange(0,1,0.01):
#    pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
#    pred=pred.iloc[:,1]
#    print (str(i)+":"+str(f1_score(y_cross, pred)))
pred = pred_proba_df.applymap(lambda x: 1 if x>0.40 else 0) #0.2-0.3
pred=pred.iloc[:,1]
#print(accuracy)
#print (f1_score(y_cross, pred, average='macro'))
print('')
"""""
log= LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_data, train_labels)
#test_loss,test_acc=network.evaluate(test_images,test_labels)
pred=log.predict(test_data)
"""""

df = pd.DataFrame(pred)
df.index+=29000
df.to_csv('/home/pedromi/Descargas/iniciacionSpainML-Credit/results.csv', index_label='id',header=['prediction'])

"""""
network=models.Sequential()
network.add(layers.Dense(50,activation='relu', input_shape=(23,1)))
network.add(layers.Dense(10))
network.add(layers.Dense(2,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_data=train_data.values
train_data=train_data.reshape((29000,23,1))

test_data=dataTest.iloc[:,1:]
test_data=test_data.values
test_data=test_data.reshape((1000,23,1))

#train_labels=to_categorical(train_labels)
#test_labels=to_categorical(test_labels)


network.fit(train_data,train_labels,epochs=15,batch_size=128)

pred=network.predict_classes(test_data)
#test_loss,test_acc=network.evaluate(test_images,test_labels)
print(pred)

df = pd.DataFrame(pred)
df.index+=29000
df.to_csv('/home/pedromi/Descargas/iniciacionSpainML-Credit/results.csv', index_label='id',header=['prediction'])
"""""