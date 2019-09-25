# OneClass SVM Gamma comparation

import time
from keras import layers
from keras import models
from keras.utils import to_categorical
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from scipy.io import arff
import pandas as pd
from os import listdir
import warnings

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


np.set_printoptions(threshold=np.inf)
""""
ficheros_sensores=listdir("Sensores2")
ficheros_sensores.sort()
data = arff.loadarff('VectorSensores.arff')
df = pd.DataFrame(data[0])
nColumnasSensores=39
df=df.iloc[:,0:nColumnasSensores]
dfOrig=df
sizeDatosUsuarioNormal = df.shape[0]

#outliers_fractions = [0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]
#gammas=[0.01,0.1,1]
#kernels=["rbf","linear"]
anomaliasTest=[]
normalTest=[]

dfAll=df
#normalized_df = (dfAll - dfAll.mean()) / dfAll.std()
#dfAll = normalized_df
#dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['diaSemana'], prefix='diaSemana')], axis=1)
#dfAll.drop(['diaSemana'], axis=1, inplace=True)
dfUtil, dfCross = train_test_split(dfAll, test_size=0.10, random_state=1)
#OCSVM = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.1)
OCSVM=IsolationForest(contamination=0.2,random_state=42)
OCSVM.fit(dfUtil)

for file in ficheros_sensores:
    print("File:",file)

    df=dfOrig
    dataEv = arff.loadarff("Sensores2/"+file)
    dfEval = pd.DataFrame(dataEv[0])
    dfEvalUtil = dfEval.iloc[:, 0:nColumnasSensores]
    dfAll=pd.concat([df,dfEvalUtil])
    #normalized_df = (dfAll - dfAll.mean()) / dfAll.std()
    #dfAll = normalized_df
    #dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['diaSemana'], prefix='diaSemana')], axis=1)
    #dfAll.drop(['diaSemana'], axis=1, inplace=True)
    #dfB=dfAll.iloc[:sizeDatosUsuarioNormal,:]
    dfEvalUtil=dfAll.iloc[sizeDatosUsuarioNormal:,:]

    y_res=OCSVM.predict(dfCross)
    unique_elements, counts_elements = np.unique(y_res, return_counts=True)
    print("OC-SVM cross validation: ",unique_elements, counts_elements)
    y_res=OCSVM.predict(dfEvalUtil)
    unique_elements, counts_elements = np.unique(y_res, return_counts=True)
    print("OC-SVM fichero test: ",unique_elements, counts_elements)
    anomaliasTest.append(counts_elements[0])
    normalTest.append(counts_elements[1])
    print('')

print (sum(anomaliasTest))
print (sum(normalTest))
listaUsuarios = range(1,21)
plt.title("Sensor data evaluation with Isolation Forest without time data")
anomaliasTestPlot=[(int)(a*100/(a+b)) for a,b in zip(anomaliasTest,normalTest)]
normalTestPlot=[(int)(a*100/(a+b)) for a,b in zip(normalTest,anomaliasTest)]
plt.bar(listaUsuarios,anomaliasTestPlot,width=0.3,color='r',label='anomalies')
plt.bar(list(x+0.3 for x in listaUsuarios),normalTestPlot,width=0.3,color='b',label='normal')
plt.xlabel("User")
plt.ylabel("Value")
plt.xticks(np.arange(1, 20+1, 1.0))
plt.legend(loc='upper right')
#plt.gca().set_ylim(0,200)
plt.show()
"""
"""
ficheros_apps=listdir("Apps3")
ficheros_apps.sort()
data = arff.loadarff('Vector.arff')
df = pd.DataFrame(data[0])
nColumnasApps=13
dfDatosUsuario=df.iloc[:,0:nColumnasApps]
dfDatosTiempo=df.iloc[:,nColumnasApps+1:]
dfOrig=df
sizeDatosUsuarioNormal = df.shape[0]
anomaliasTest=[]
normalTest=[]
sizeDatosEvalUsuarios=[]
#dfAll=dfDatosUsuario
dfAll=pd.concat([dfDatosUsuario,dfDatosTiempo],axis=1)
dfAllRes=df.iloc[:,13:14]

cont=2
for file in ficheros_apps:
    dataEv = arff.loadarff("Apps3/"+file)
    dfEval = pd.DataFrame(dataEv[0])
    sizeDatosEval=dfEval.shape[0]
    sizeDatosEvalUsuarios.append(sizeDatosEval)
    dfEvalUtil = dfEval.iloc[:, 0:13]
    dfEvalTiempo=dfEval.iloc[:,14:]
    dfGenRes=dfEval.iloc[:,13:14]
    dfGenRes=dfGenRes*cont
    #dfEval=dfEvalUtil
    dfEval=pd.concat([dfEvalUtil,dfEvalTiempo],axis=1)
    dfAll=pd.concat([dfAll,dfEval])
    dfAllRes=pd.concat([dfAllRes,dfGenRes])
    cont+=1

dfAllRes=LabelEncoder().fit_transform(dfAllRes)

normalized_df = (dfAll - dfAll.mean()) / dfAll.std()
dfAll = normalized_df
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['appMasUsadaUltimoMinuto'], prefix='appMasUsadaUltimoMinuto')], axis=1)
dfAll.drop(['appMasUsadaUltimoMinuto'], axis=1, inplace=True)
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['ultimaApp'], prefix='ultimaApp')], axis=1)
dfAll.drop(['ultimaApp'], axis=1, inplace=True)
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['anteriorUltimaApp'], prefix='anteriorUltimaApp')], axis=1)
dfAll.drop(['anteriorUltimaApp'], axis=1, inplace=True)
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['appMaxAnterior'], prefix='appMaxAnterior')], axis=1)
dfAll.drop(['appMaxAnterior'], axis=1, inplace=True)
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['appMasUsadaUltimoDia'], prefix='appMasUsadaUltimoDia')], axis=1)
dfAll.drop(['appMasUsadaUltimoDia'], axis=1, inplace=True)
dfAll = pd.concat([dfAll, pd.get_dummies(dfAll['diaSemana'], prefix='diaSemana')], axis=1)
dfAll.drop(['diaSemana'], axis=1, inplace=True)

dfUsuario= dfAll.iloc[:sizeDatosUsuarioNormal, :]
datosEvalSeparados=[]
dfEvalUtil = dfAll.iloc[sizeDatosUsuarioNormal:, :]
contador=sizeDatosUsuarioNormal
for size in sizeDatosEvalUsuarios:
    datosEvalSeparados.append(dfAll.iloc[contador:contador+size,:])
    contador+=size
"""
"""
rf = RandomForestRegressor()
rf.fit(dfAll, dfAllRes)
names = dfAll.columns.values
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True))
"""
"""
X_train,X_cross, y_train,y_cross = train_test_split(dfAll,dfAllRes,stratify=dfAllRes, test_size=0.10, random_state=1)
clasif=svm.SVC(gamma=0.2)
clasif.fit(X_train,y_train)
pred=clasif.predict(X_cross)
accuracy=clasif.score(X_cross,y_cross)
print(accuracy)
print (f1_score(y_cross, pred, average='macro'))


dfUtil, dfCross = train_test_split(dfUsuario, test_size=0.10, random_state=1)
OCSVM = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.1)
#OCSVM=IsolationForest(contamination=0.2,random_state=42)
OCSVM.fit(dfUtil)
puntuaciones=OCSVM.decision_function(dfCross)
#Histograma puntuaciones
print(puntuaciones)
print(puntuaciones.max())
print(puntuaciones.min())
#outliers_fractions = [0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]
#gammas=[0.0001,0.001,0.01,0.1,1,10,100]
#kernels=["rbf","linear"]
##0:13
#for file in ficheros_apps:
    #print("File ",file)
y_res=OCSVM.predict(dfCross)
unique_elements, counts_elements = np.unique(y_res, return_counts=True)
print("OC-SVM cross validation: ",unique_elements, counts_elements)
y_res=OCSVM.predict(dfEvalUtil)
unique_elements, counts_elements = np.unique(y_res, return_counts=True)
print("OC-SVM todos usuarios: ",unique_elements, counts_elements)
print('')
anomaliasTest=[]
normalTest=[]

for i in range(0,len(datosEvalSeparados)):
    y_res = OCSVM.predict(datosEvalSeparados[i])
    unique_elements, counts_elements = np.unique(y_res, return_counts=True)
    print(str(ficheros_apps[i])+": ", unique_elements, counts_elements)
    anomaliasTest.append(counts_elements[0])
    if(len(counts_elements)!=1):
        normalTest.append(counts_elements[1])
    else:
        normalTest.append(0)
    print('')

print (sum(anomaliasTest))
print (sum(normalTest))
listaUsuarios = range(1,21)
plt.title("Application data evaluation with OC-SVM including time data without pre-processing")
anomaliasTestPlot=[(int)(a*100/(a+b)) for a,b in zip(anomaliasTest,normalTest)]
normalTestPlot=[(int)(a*100/(a+b)) for a,b in zip(normalTest,anomaliasTest)]
plt.bar(listaUsuarios,anomaliasTestPlot,width=0.3,color='r',label='anomalies')
plt.bar(list(x+0.3 for x in listaUsuarios),normalTestPlot,width=0.3,color='b',label='normal')
plt.xlabel("User")
plt.ylabel("Value")
plt.xticks(np.arange(1, 20+1, 1.0))
plt.legend(loc='upper right')
#plt.gca().set_ylim(0,200)
plt.show()


"""


ficheros_apps=listdir("Apps3")
ficheros_apps.sort()
datosUsuarios=pd.DataFrame()
etiquetasUsuarios=pd.DataFrame()

for file in ficheros_apps:
    dataEv = arff.loadarff("Apps3/"+file)
    dataEv=pd.DataFrame(dataEv[0])
    dataEv["Etiqueta"]=file
    etiqEv=dataEv["Etiqueta"]
    dataEv = dataEv.drop("Etiqueta", axis=1)
    datosUsuarios = pd.concat([datosUsuarios, dataEv])
    etiquetasUsuarios=pd.concat([etiquetasUsuarios,etiqEv])

print(datosUsuarios.head())

le=LabelEncoder()
datos_usuarios_etiquetas=le.fit_transform(etiquetasUsuarios)

print(datos_usuarios_etiquetas)

#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['appMasUsadaUltimoMinuto'], prefix='appMasUsadaUltimoMinuto')], axis=1)
#datosUsuarios.drop(['appMasUsadaUltimoMinuto'], axis=1, inplace=True)
#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['ultimaApp'], prefix='ultimaApp')], axis=1)
#datosUsuarios.drop(['ultimaApp'], axis=1, inplace=True)
#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['anteriorUltimaApp'], prefix='anteriorUltimaApp')], axis=1)
#datosUsuarios.drop(['anteriorUltimaApp'], axis=1, inplace=True)
#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['appMaxAnterior'], prefix='appMaxAnterior')], axis=1)
#datosUsuarios.drop(['appMaxAnterior'], axis=1, inplace=True)
#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['appMasUsadaUltimoDia'], prefix='appMasUsadaUltimoDia')], axis=1)
#datosUsuarios.drop(['appMasUsadaUltimoDia'], axis=1, inplace=True)
#datosUsuarios = pd.concat([datosUsuarios, pd.get_dummies(datosUsuarios['diaSemana'], prefix='diaSemana')], axis=1)
datosUsuarios.drop(['diaSemana'], axis=1, inplace=True)
datosUsuarios.drop(['horaActual'], axis=1, inplace=True)
datosUsuarios.drop(['usuario'],axis=1,inplace=True)
print(datosUsuarios.head())

X_train,X_test, y_train,y_test = train_test_split(datosUsuarios,datos_usuarios_etiquetas,stratify=datos_usuarios_etiquetas, test_size=0.40, random_state=2)


rf = RandomForestClassifier(n_estimators = 100,random_state=1)

rf.fit(X_train, y_train)


#random_forest_file=open("/home/pedromi/DatosTFM/random_forest_file.pickle",'wb')
#pickle.dump(rf,random_forest_file)
#random_forest_file.close()


pred=rf.predict(X_test)
accuracy=rf.score(X_test,y_test)
print("Accuracy: {}".format(accuracy))

precision, recall, fscore, support = score(y_test, pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

feat_labels=X_train.columns
importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))