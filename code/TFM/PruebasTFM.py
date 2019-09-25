import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from scipy.io import arff
import pandas as pd
from os import listdir
import warnings

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
import time
import pickle
from sklearn.metrics import precision_recall_fscore_support as score

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

"""
##########
##########CREAR FICHERO CON TODOS LOS DATOS
##########

ficheros_usuarios=listdir("/home/pedromi/DatosTFM/archivos/")
ficheros_usuarios.sort()
datos_usuarios=pd.DataFrame()

for file in ficheros_usuarios:
    data=pd.read_csv("/home/pedromi/DatosTFM/archivos/"+file,header=None)
    if file=="mgilperez":
        data["Etiqueta"]="ManuelGP"
    else:
        data["Etiqueta"]=file
    print(data.head())
    print(data.shape[0])
    #datos_usuarios=pd.concat([datos_usuarios,data])
    if os.path.exists("/home/pedromi/DatosTFM/datos_usuarios"):
        data.to_csv("/home/pedromi/DatosTFM/datos_usuarios",mode='a', header=None, index=None)
    else:
        data.to_csv("/home/pedromi/DatosTFM/datos_usuarios", index=None, header=None)
"""
"""
##########
########### CAMBIAR MILISEGUNDOS POR HORAS
##########
datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios")
print(datos_usuarios.shape[1])
for i in range(0,datos_usuarios["0"].shape[0]):
    datos_usuarios["0"][i]=time.localtime(int(datos_usuarios["0"][i]/1000)).tm_hour
datos_usuarios.to_csv("/home/pedromi/DatosTFM/datos_usuarios_horadia",index=None)
print(datos_usuarios.head())
"""
######
##### BORRAR COLUMNAS IGUAL A CERO
#####
# datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_cabecera")
# datos_usuarios=datos_usuarios.loc[:, (datos_usuarios != 0).any(axis=0)]
# datos_usuarios.to_csv("/home/pedromi/DatosTFM/datos_usuarios_sin_columnas_cero",index=None)


####
### SEPARAR RATON Y TECLADO
###
"""
datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_sin_columnas_cero")
etiquetas=datos_usuarios["ETIQUETA"]
datos_usuarios=datos_usuarios.iloc[:,:-18]
datos_usuarios=pd.concat([datos_usuarios,etiquetas],axis=1)
print(datos_usuarios.head())
datos_usuarios.to_csv("/home/pedromi/DatosTFM/datos_usuarios_raton_teclado",index=None)
"""

####
### SEPARAR APLICACIONES y ONE HOT
###
"""
datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_sin_columnas_cero")
tiempo=datos_usuarios["MarcaTiempo"]
datos_usuarios=datos_usuarios.iloc[:,-18:]
datos_usuarios=pd.concat([datos_usuarios,tiempo],axis=1)
datos_usuarios = pd.concat([datos_usuarios, pd.get_dummies(datos_usuarios['aplicacion_actual'], prefix='aplicacion_actual')], axis=1)
datos_usuarios.drop(['aplicacion_actual'], axis=1, inplace=True)
datos_usuarios = pd.concat([datos_usuarios, pd.get_dummies(datos_usuarios['penultima_aplicacion'], prefix='penultima_aplicacion')], axis=1)
datos_usuarios.drop(['penultima_aplicacion'], axis=1, inplace=True)
print(datos_usuarios.head())
datos_usuarios.to_csv("/home/pedromi/DatosTFM/datos_usuarios_aplicaciones",index=None)
"""

####
### ONE HOT ENCONDIG APPS
###
"""
datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_sin_columnas_cero")

datos_usuarios = pd.concat([datos_usuarios, pd.get_dummies(datos_usuarios['aplicacion_actual'], prefix='aplicacion_actual')], axis=1)
datos_usuarios.drop(['aplicacion_actual'], axis=1, inplace=True)
datos_usuarios = pd.concat([datos_usuarios, pd.get_dummies(datos_usuarios['penultima_aplicacion'], prefix='penultima_aplicacion')], axis=1)
datos_usuarios.drop(['penultima_aplicacion'], axis=1, inplace=True)

datos_usuarios.to_csv("/home/pedromi/DatosTFM/datos_one_hot",index=None)
"""

######
###### COMBIERTE MILLIS A HORA, HACE MAX, MIN Y GUARDA ETIQUETAS Y DATOS POR SEPARADO
#####
"""
datos_usuarios=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios")
print("HORAS")
for i in range(0,datos_usuarios["MarcaTiempo"].shape[0]):
    datos_usuarios["MarcaTiempo"][i]=time.localtime(int(datos_usuarios["MarcaTiempo"][i]/1000)).tm_hour

datos_usuarios_etiquetas=datos_usuarios["ETIQUETA"]
datos_usuarios_valores=datos_usuarios.drop("ETIQUETA",axis=1)
#datos_usuarios_valores=(datos_usuarios_valores-datos_usuarios_valores.min())/(datos_usuarios_valores.max()-datos_usuarios_valores.min())
print("ESCRIBIR")
#datos_usuarios_etiquetas.to_csv("/home/pedromi/DatosTFM/datos_usuarios_etiquetas",index=None)
datos_usuarios_valores.to_csv("/home/pedromi/DatosTFM/datos_usuarios_valores",index=None)
"""

###
### PRUEBAS RANDOM FOREST IMPORTANCIA FEATURES Y CLASIFICACION
###
"""
datos_usuarios_etiquetas=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_etiquetas")
print(datos_usuarios_etiquetas.shape[0])
datos_usuarios_valores=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_valores")
print(datos_usuarios_valores.shape[0])

datos=pd.concat([datos_usuarios_valores,datos_usuarios_etiquetas],axis=1)
datos=datos[datos.ETIQUETA.isin(["FRS","Gregorio","PedroASr1","ManuelGP","pedro"])]
#datos=datos[datos.ETIQUETA.isin(["FRS","slopez","PedroASr1","ManuelGP","pedro"])]
datos_usuarios_etiquetas=datos["ETIQUETA"]
datos_usuarios_valores=datos.drop("ETIQUETA",axis=1)
le=LabelEncoder()
datos_usuarios_etiquetas=le.fit_transform(datos_usuarios_etiquetas)
X_train,X_test, y_train,y_test = train_test_split(datos_usuarios_valores,datos_usuarios_etiquetas,stratify=datos_usuarios_etiquetas, test_size=0.10, random_state=2)

#label_file=open("/home/pedromi/DatosTFM/label_encoder.pickle",'wb')
#pickle.dump(le,label_file)
#label_file.close()

#max=X_train.max()
#min=X_train.min()

#X_train=(X_train-min)/(max-min)
#X_test=(X_test-min)/(max-min)

#X_train=X_train.fillna(0)
#X_train=X_train.replace([np.inf, -np.inf], 0)
#X_test=X_test.fillna(0)
#X_test=X_test.replace([np.inf, -np.inf], 0)
print("Entrena")
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
"""

"""
import xgboost as xgb
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 14}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

X_train.columns = range(len(X_train.columns.values))
X_test.columns = range(len(X_test.columns.values))


max=X_train.max()
min=X_train.min()

X_train=(X_train-min)/(max-min)
X_test=(X_test-min)/(max-min)

X_train=X_train.fillna(0)
X_train=X_train.replace([np.inf, -np.inf], 0)
X_test=X_test.fillna(0)
X_test=X_test.replace([np.inf, -np.inf], 0)

xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_test)
bst = xgb.train(param, xgtrain, num_round)
pred = bst.predict(xgtest)
precision, recall, fscore, support = score(y_test, pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
"""
"""
feat_labels=X_train.columns
importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

"""

"""
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
#plt.tight_layout()
plt.show()
"""

######
###### Seleccionar features
######
"""
random_forest_file=open("/home/pedromi/DatosTFM/random_forest_file.pickle",'rb')
random_forest=pickle.load(random_forest_file)
random_forest_file.close()


importances=random_forest.feature_importances_
indices=np.argsort(importances)[::-1]
i=0
peso=0
while(peso<0.99):
    peso+=importances[indices[i]]
    i=i+1
print(i+1)


"""
"""
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

from sklearn.feature_selection import SelectFromModel

sfm=SelectFromModel(random_forest,prefit=True)


X_Selected=sfm.transform(X_train)
print(X_Selected.shape[1])
for f in range(X_Selected.shape[1]):
    print("%2d) %-*s %f"%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))
"""

##########
########## INFORMACION MUTUA, FEATURES Y ETIQUETA
##########
"""
import sklearn
informacion=sklearn.feature_selection.mutual_info_.mutual_info_classif(X_train,y_train)
print(informacion)
print(max(informacion))
for score, fname in sorted(zip(informacion, X_train.columns.values), reverse=True)[:]:
    print(fname, score)
"""

######
###### CORRELATION MATRIX
######
"""
import seaborn as sns
corr=X_train.corr()
plt.matshow(corr)
plt.show()

sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
"""

##########
########## TEST DETECCION ANOMALIAS
##########
"""
datos_usuarios_etiquetas=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_etiquetas")
print(datos_usuarios_etiquetas.shape[0])
datos_usuarios_valores=pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_valores")
#datos_usuarios_valores.drop("ETIQUETA",1,inplace=True)
print(datos_usuarios_valores.shape[0])
#datos_usuarios_valores=datos_usuarios_valores.drop("ETIQUETA",axis=1)
#le=LabelEncoder()
#datos_usuarios_etiquetas=le.fit_transform(datos_usuarios_etiquetas)

datos_usuarios=pd.concat([datos_usuarios_valores,datos_usuarios_etiquetas],axis=1)
#X_train,X_test, y_train,y_test = train_test_split(datos_usuarios_valores,datos_usuarios_etiquetas,stratify=datos_usuarios_etiquetas, test_size=0.10, random_state=2)

for nombre in datos_usuarios_etiquetas.ETIQUETA.unique():
    print("Deteccion anomalías para "+nombre)
    datos_otros_usuarios=datos_usuarios.loc[datos_usuarios['ETIQUETA'] != nombre]
    datos_otros_usuarios.drop("ETIQUETA",1,inplace=True)
    datos_usuario=datos_usuarios.loc[datos_usuarios['ETIQUETA'] == nombre]
    datos_usuario.drop("ETIQUETA",1,inplace=True)
    X_test=datos_usuario

    X_train,X_test=train_test_split(datos_usuario, test_size=0.10, random_state=1)

    from sklearn import svm
    #clf = LocalOutlierFactor(n_neighbors=35,contamination=0.1)
    clf = IsolationForest(behaviour="old",contamination=0.4)
    #clf=svm.OneClassSVM(kernel='rbf',gamma=10)
    clf.fit(X_train)
    X_train_otros,X_test_otros=train_test_split(datos_otros_usuarios, test_size=0.10)
    X_test_otros=X_test_otros.iloc[:int(len(X_test)),:]
    numero_datos_otros_usuarios=len(X_test_otros)
    numero_datos_usuario=len(X_test)

    X_test=pd.concat([X_test,X_test_otros])
    X_test=(X_test-X_test.mean())/(X_test.std())
    X_test=X_test.replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0)
    res=clf.predict(X_test)
    unique_elements, counts_elements = np.unique(res, return_counts=True)
    print(unique_elements,"    ",counts_elements)
    contador_usuario=[]
    contador_otros_usuarios=[]
    for i in range(len(res)):
        if i <= numero_datos_usuario-1:
            contador_usuario.append(res[i])
        else:
            contador_otros_usuarios.append(res[i])

    unique_elements, counts_elements = np.unique(contador_usuario, return_counts=True)
    print("Datos usuario: ",unique_elements,"    ",counts_elements, " Numero total de vectores: ",numero_datos_usuario)
    unique_elements, counts_elements = np.unique(contador_otros_usuarios, return_counts=True)
    print("Datos otros usuarios: ",unique_elements,"    ",counts_elements, " Numero total de vectores: ",numero_datos_otros_usuarios)
"""

########################## EDA #################
print("EDA")
datos_usuarios = pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_valores")
datos_usuarios_etiquetas = pd.read_csv("/home/pedromi/DatosTFM/datos_usuarios_etiquetas")

datos_usuarios["USER"] = datos_usuarios_etiquetas["ETIQUETA"]

import matplotlib.pyplot as plt

users = datos_usuarios["USER"].unique()

######HISTOGRAMA PULSACIONES TECLAS GENERAL####
"""
filter_col = [col for col in datos_usuarios if col.startswith('pulsacion_digrafo_')]
datos_usuarios = datos_usuarios[filter_col]
datos_usuarios = datos_usuarios.rename(columns=lambda x: str(x)[18:])
suma = datos_usuarios.sum(axis=0, skipna=True)
suma = suma.sort_values(ascending=False)
suma = suma.head(100)
print(suma)
#suma.to_csv("/home/pedromi/AutenticacionContinuaPC_paper/EDA/Histogramas_pulsaciones_teclas/100_mas_usadas.csv")
suma.plot.bar()
plt.title("Histograma Teclas Mas Usadas")
plt.show()
"""


####HISTOGRAMA PULSACIONES TECLAS POR USUARIO###
"""
filter_col = [col for col in datos_usuarios if col.startswith('pulsacion_digrafo_')]
filter_col.append("USER")
datos_usuarios = datos_usuarios[filter_col]

for usuario in users:
    datos_usuario_actual=datos_usuarios.loc[datos_usuarios['USER'] == usuario]
    datos_usuario_actual = datos_usuario_actual.drop('USER', 1)
    datos_usuario_actual = datos_usuario_actual.rename(columns=lambda x: str(x)[18:])
    suma = datos_usuario_actual.sum(axis=0, skipna=True)
    suma = suma.sort_values(ascending=False)
    suma = suma.head(100)
    print("Teclas mas usadas por "+usuario)
    print(suma)
    suma.plot.bar()
    plt.title("Histograma Teclas Mas Usadas por "+usuario)
    plt.show()
"""


"""
#####HISTOGRAMAS POR HORAS####
filter_col = [col for col in datos_usuarios if col.startswith('pulsacion_digrafo_')]
filter_col.append("USER")
datos_usuarios = datos_usuarios[filter_col]

for usuario in users:
    horas=datos_usuarios.loc[datos_usuarios['USER'] == usuario]
    horas=horas["MarcaTiempo"].values
    plt.hist(horas, bins = 24,range=[0,23])
    plt.title("Histograma 24H "+str(usuario))
    plt.show()
"""
