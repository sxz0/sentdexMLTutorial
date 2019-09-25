def histogramaPalabras(lista):
    histograma = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
    for n in lista:
        if n >= 11:
            histograma[11] += 1
        else:
            histograma[n] += 1
    return histograma

def histogramaPosicion(lista):
    histograma = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for n in lista:
        histograma[n] += 1
    return histograma
def media(lista):
    if len(lista) == 0:
        return 0
    return round(sum(lista) / len(lista), 2)


def calcularMediaListasDiccionario(diccionario):
    diccionario_resultado = {}
    for k in diccionario.keys():
        diccionario_resultado[k] = media(diccionario[k])
    return diccionario_resultado

def histogramaDireccion(lista):
    histograma = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    for n in lista:
        histograma[n] += 1
    return histograma


def histogramaLongitudMovimiento(lista):
    histograma = {1: 0, 2: 0, 3: 0}
    for n in lista:
        histograma[n] += 1
    return histograma


"""
CONSTANTES
"""
CONSTANTE_DOBLE_CLICK = 500
CONSTANTE_SUBVENTANA = 500
""""
CONTADORES EVENTOS
"""
eventos_teclado = 0
eventos_raton = 0
""""
VARIABLES PARA EL TECLADO
"""
lista_caracteres = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                    'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ç']
lista_espaciado = ['-', '_', '.', ',', '/', '&', '+', '<', 'space', 'tab', 'enter', '(', ')', '=', '|', '\\', '#']
longitud_palabra = 0  # logitud de la palabra actual
numero_palabras = 0  # numero de palabras en la ventana actual
lista_longitud_palabras = []
lista_todas_teclas = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                      'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '´',
                      '`', "'", '"', 'ç', '^', 'º', '@', '$', '%', '&', '/', '(', ')', '=', '|', 'windowsizquierda',
                      'crtl',
                      'mayusculas', 'bloqmayus', 'tab', 'º', 'ª', '\\', '#', 'esc', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                      'f7', 'f8',
                      'f9', 'f10', 'f11', 'f12', 'imppant', 'insert', 'supr', 'inicio', 'fin', 'repag', 'avpag',
                      'numlock',
                      '}', '{', '-', '_', '.', ',', '[', ']', '*', '<', '>', 'space', 'tab', 'enter', 'ctrlderecha',
                      'rightshift', 'backspace', 'atlgr', 'alt', 'left', 'right', 'up', 'down', 'flechaderecha',
                      'flechaizquierda', 'flechaarriba', 'flechaabajo', '+']

marca_tiempo = 0
ultima_tecla_pulsada = ''
marca_tiempo_ultima_pulsacion = 0  # Marca de tiempo de la ultima tecla pulsada
total_teclas_presionadas = 0  # Numero de teclas presionadas en la ventana de tiempo
total_teclas_borrado = 0  # Numero de teclas de borrado presionadas
pulsaciones_por_tecla = {}  # Numero de pulsaciones por tecla
for t in lista_todas_teclas:
    pulsaciones_por_tecla[t] = 0
teclas_presionadas = []  # Teclas presionadas
intervalo_tiempo_presionar_teclas = []  # Intervalos de tiempo en orden de pulsacion de teclas
lista_intervalos_pulsar_soltar_tecla = []  # Intervalos de pulsar y soltar cada tecla
marcas_tiempo_presionado = {}  # Marca de tiempo cuando cada letra ha sido presionada por ultima vez
for t in lista_todas_teclas:
    marcas_tiempo_presionado[t] = 0
intervalos_pulsar_soltar_por_tecla = {}  # Lista de tiempo de pulsaciones de cada tecla
for t in lista_todas_teclas:
    intervalos_pulsar_soltar_por_tecla[t] = []
intervalo_digrafo = {}  # tiempo entre dos pulsaciones de todas las teclas
for t in lista_todas_teclas:
    for t2 in lista_todas_teclas:
        intervalo_digrafo[t + t2] = []
digrafo = {}  # pulsaciones de secuencia de dos teclas
for t in lista_todas_teclas:
    for t2 in lista_todas_teclas:
        digrafo[t + t2] = 0

"""
VARIABLES PARA RATON
"""
marcas_tiempo_click = {0: 0, 1: 0, 3: 0}
intervalo_tiempo_click = {0: [], 1: [], 2: [], 3: []}
lista_posicion_raton = []
# Última posición X
marca_x = 0
# Última posición Y
marca_y = 0
# Lista con las direcciones de los movimientos
lista_direcciones_movimiento = []
# Posición del primer evento de movimiento dentro de la subventana
marca_x_angulo = 0
marca_y_angulo = 0
# Lista para las distintas acciones del ratón
lista_acciones_raton = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

# Auxiliar para controlar cuando se está arrastrando
arrastrar = False
# Auxiliar para controlar el último movimiento (útil para la duración periodo pausa de movimiento click)
ultima_accion_raton = ''
# Marca del tiempo del último movimiento para obtener la duración media del tiempo de pausa movimiento-click
marca_tiempo_movimiento = 0
# Duración periodo pausa movimiento-click
intervalo_movimiento_click = []
# Longitud del movimiento actual
longitud_recorrida = 0
# Lista longitud de movimientos
lista_longitud_movimiento = []
# Lista con las velocidades medias
lista_velocidades_medias = []
# Lista con las velocidades medias en las 8 direcciones
lista_velocidades_medias_direcciones = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
# Booleano auxiliar para controlar si es doble click o no
esDobleClick = False
# Variables para calcular la velocidad con el tiempo del primer y el último movimiento
marca_tiempo_primer_movimiento = 0
# Auxilar para saber si el click anterior era el segundo de un doble click
anteriorDobleClick = True
"""
VARIABLES APPS
"""
lista_numero_apps = []
lista_nombre_apps = []
cpu_por_proceso = {}
ram_por_aplicacion = {}
lista_cpu_total = []
lista_ram_total = []
bytes_recv = 0
bytes_sent = 0
tiempo_primer_plano_en_ventana = {}
procesos_por_aplicacion = {}
"""
VARIABLES AUXILIARES
"""
inicio_subventana = 0


cabecera=""
cabecera+="MarcaTiempo,"
cabecera+="total_teclas_presionadas,"
cabecera+="total_teclas_borrado,"
cabecera+="proporcion_teclas_borrado,"
cabecera+="media_intervalo_presionar_teclas,"
cabecera+="desviacion_intervalo_presionar_teclas,"
cabecera+="media_intervalo_presionar_soltar,"
cabecera+="desviacion_intervalo_presionar_soltar,"
cabecera+="numero_palabras,"
cabecera+="media_longitud_palabras,"
cabecera+="desviacion_longitud_palabras,"
for v in histogramaPalabras(lista_longitud_palabras).keys():
    cabecera+="palabras_longitud_"+str(v)+","

for v in pulsaciones_por_tecla.keys():
    v=v.replace(",","coma")
    cabecera += "numero_pulsaciones_"+str(v)+","

for v in intervalos_pulsar_soltar_por_tecla.keys():
    v=v.replace(",","coma")
    cabecera += "media_pulsar_soltar_"+str(v)+","

for v in digrafo.keys():
    v=v.replace(",","coma")
    cabecera += "pulsacion_digrafo_"+str(v)+","

for v in digrafo.keys():
    v=v.replace(",","coma")
    cabecera += "media_pulsacion_digrafo_"+str(v)+","

for v in range(0,4):
    cabecera += "media_intervalo_click_"+str(v)+","

for v in range(0,4):
    cabecera += "desviacion_intervalo_click_"+str(v)+","

for v in lista_acciones_raton.keys():
    cabecera+="accion_raton_"+str(v)+","

for v in histogramaPosicion(lista_posicion_raton).keys():
    cabecera+="histograma_posicion_"+str(v)+","

for v in histogramaDireccion(lista_direcciones_movimiento).keys():
    cabecera+="histograma_direccion_"+str(v)+","

for v in histogramaLongitudMovimiento(lista_longitud_movimiento).keys():
    cabecera+="histograma_logitud_"+str(v)+","

cabecera+="media_intervalo_movimiento,"
cabecera += "media_velocidad_movimiento,"

for v in lista_velocidades_medias_direcciones.keys():
    cabecera+="velocidad_media_"+str(v)+","

cabecera += "media_numero_apps,"
cabecera += "aplicacion_actual,"
cabecera += "penultima_aplicacion,"
cabecera += "numero_cambios_aplicacion,"
cabecera += "tiempo_primer_plano_ultima_aplicacion,"
cabecera += "media_numero_procesos_aplicacion_actual,"
cabecera += "desviacion_numero_procesos_aplicacion_actual,"
cabecera += "media_cpu_aplicacion_actual,"
cabecera += "desviacion_cpu_aplicacion_actual,"
cabecera += "media_cpu_total,"
cabecera += "desviacion_cpu_total,"
cabecera += "media_ram_aplicacion_actual,"
cabecera += "desviacion_ram_aplicacion_actual,"
cabecera += "media_ram_total,"
cabecera += "desviacion_ram_total,"
cabecera += "bytes_recibidos,"
cabecera += "bytes_enviados,"
cabecera += "ETIQUETA\n"

fich=open("cabecera","w")
fich.write(cabecera)
fich.close()

print(len(cabecera.split(",")))
