import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from os import listdir
import warnings

colores={'VectorAlberto.arff':'b','VectorAlberto6.arff':'g','VectorAlberto7.arff':'r','VectorFarina.arff':'c','VectorGregorio.arff':'m','VectorMora.arff':'y'}
warnings.filterwarnings("ignore")
dias=[3355393,3273460,3344016,3619420]
ficheros_sensores=listdir("Apps3")
ficheros_sensores.sort()
for dia in dias:
    for file in ficheros_sensores:
        data = arff.loadarff("Apps3/" + file)
        df = pd.DataFrame(data[0])
        sizeDatosEval = df.shape[0]
        print(sizeDatosEval)
        dfDia=df.loc[df['diaSemana']==dia]
        print(dfDia.head())
        dfDia=dfDia.sort_values('horaActual')
        print(dfDia.head())

        color=colores[file]
        ax1=plt.subplot(611)
        plt.plot(dfDia['horaActual'],dfDia['appAbiertasMinuto'],color+'o',alpha=0.3)
        plt.title('appAbiertasMinuto')
        ax1.set_xticklabels([])


        ax2=plt.subplot(612,sharex=ax1)
        plt.plot(dfDia['horaActual'],dfDia['appAbiertasDistintasMinuto'],color+'o',alpha=0.3)
        plt.title('appAbiertasDistintasMinuto')

        ax3 = plt.subplot(613, sharex=ax1)
        plt.plot(dfDia['horaActual'], dfDia['appMasUsadaUltimoMinuto'],color+'o',alpha=0.3)
        plt.title('appMasUsadaUltimoMinuto')

        ax4 = plt.subplot(614, sharex=ax1)
        plt.plot(dfDia['horaActual'], dfDia['numeroVecesUltimoMinuto'],color+'o',alpha=0.3)
        plt.title('numeroVecesUltimoMinuto')

        ax5 = plt.subplot(615, sharex=ax1)
        plt.plot(dfDia['horaActual'], dfDia['bytesRecibidosMinuto'],color+'o',alpha=0.3)
        plt.title('bytesRecibidosMinuto')

        ax6 = plt.subplot(616)
        plt.plot(dfDia['horaActual'], dfDia['bytesEnviadosMinuto'],color+'o',alpha=0.3)
        plt.title('bytesEnviadosMinuto')

    plt.show()