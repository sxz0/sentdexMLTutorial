import os

import numpy as np
import pandas as pd
from os import listdir

"""
###Obtener marcas tiempo PC y marca nombre dispositivo PC
user="mattia"
f_out=open(user+'_pc_procesado','a')
with open(user+'_pc') as f:
    for linea in f:
        troceado=linea.split(',')
        f_out.write(troceado[0]+",pc\n")

f_out.close()

##Obtener marcas tiempo movil y dimension

f_out=open(user+'_movil_procesado','a')
with open(user+'_movil','r', encoding="ISO-8859-1") as f:
    for linea in f:
        troceado=linea.split(',')
        if(len(troceado)==16):
            f_out.write(troceado[0] + ",apps\n")
        #else:
        #    f_out.write(troceado[0]+",sensors\n")

f_out.close()

#Mezclar archivos
f_out=open(user+'_mezclado','a')
for nombre in [user+'_pc_procesado',user+'_movil_procesado']:
    with open(nombre, 'r', encoding="ISO-8859-1") as f:
        for linea in f:
            f_out.write(linea)
f_out.close()

#Abrir como csv y guardar ordenado
data=pd.read_csv(user+"_mezclado",header=None)
data=data.sort_values(0)
data.to_csv(user+"_ordenado",header=None, index=None)
"""

#PROCESAR ARCHIVOS VECTORES
"""
f_out=open('vectores_6h','a')
marca_inicio=0
marca_tiempo=0
marca_tiempo_anterior=0
ventana=21600000
f=open(user+'_ordenado','r')
n_total_pc=0
n_total_movil=0
cambio_pc_movil=0
cambio_movil_pc=0
dispositivo_anterior=""
dispositivos_mismo_tiempo=0

for linea in f:
    troceado = linea.split(',')
    marca_actual=int(troceado[0])
    dispositivo_actual=troceado[1]
    if marca_tiempo==0:
        marca_tiempo=marca_actual
    if marca_tiempo+ventana<marca_actual:
        f_out.write(str(n_total_pc)+","+str(n_total_movil)+","+str(cambio_pc_movil)+","+str(cambio_movil_pc)+","+str(dispositivos_mismo_tiempo)+","+user+"\n")
        n_total_pc = 0
        n_total_movil = 0
        cambio_pc_movil = 0
        cambio_movil_pc = 0
        dispositivo_anterior = ""
        dispositivos_mismo_tiempo = 0
        marca_tiempo=marca_actual
    else:
        if dispositivo_anterior=="":
            dispositivo_anterior=dispositivo_actual
        if dispositivo_actual=="apps\n":
            n_total_movil+=1
            if dispositivo_anterior=="pc\n":
                if marca_actual-60000<marca_tiempo_anterior:
                    dispositivos_mismo_tiempo+=1
                else:
                    cambio_pc_movil+=1

        if dispositivo_actual=="pc\n":
            n_total_pc+=1
            if dispositivo_anterior=="apps\n":
                if marca_actual-60000<marca_tiempo_anterior:
                    dispositivos_mismo_tiempo+=1
                else:
                    cambio_movil_pc += 1

        dispositivo_anterior=dispositivo_actual
        marca_tiempo_anterior=marca_actual
        
f_out.close()
"""

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

for n in [3,6,12,24]:
    data=pd.read_csv("vectores_"+str(n)+"h",header=None)

    datos_usuarios_etiquetas=data[5]
    datos_usuarios_valores=data.drop(5,axis=1)
    X_train,X_test, y_train,y_test = train_test_split(datos_usuarios_valores,datos_usuarios_etiquetas,stratify=datos_usuarios_etiquetas, test_size=0.20, random_state=2)

    rf = RandomForestClassifier(n_estimators = 100,random_state=1)
    rf = svm.SVC(gamma=0.0001)

    rf.fit(X_train, y_train)
    pred=rf.predict(X_test)
    accuracy=rf.score(X_test,y_test)

    print(str(n)+" Horas:")
    print("Accuracy: {}".format(accuracy))

    precision, recall, fscore, support = score(y_test, pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print("\n")
