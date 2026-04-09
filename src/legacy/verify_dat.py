# -*- coding: utf-8 -*-
"""
Script para leer y verificar un archivo de datos binario (.dat) que contiene
registros neuronales.

Este script comprueba la integridad del archivo, reconstruye la forma de los datos
y visualiza una porción para una inspección rápida.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def verificar_dat(filepath, num_canales, dtype=np.int16, freq_muestreo=30000):
    """
    Carga, analiza y visualiza un archivo .dat.

    Args:
        filepath (str): Ruta al archivo .dat a verificar.
        num_canales (int): El número de canales que se guardaron en el archivo.
                           Este es un metadato CRÍTICO.
        dtype (numpy.dtype, optional): El tipo de dato con el que se guardó el archivo.
                                       Por defecto es np.int16 (short).
        freq_muestreo (int, optional): La frecuencia de muestreo para el eje de tiempo.
                                       Por defecto es 30000 Hz.
    """
    # --- 1. Verificación del archivo ---
    print(f"Verificando el archivo: {filepath}")

    if not os.path.exists(filepath):
        print(f"ERROR: El archivo no se encuentra en la ruta especificada.")
        return

    # Obtenemos el tamaño del archivo en bytes
    bytes_totales = os.path.getsize(filepath)
    bytes_por_muestra = np.dtype(dtype).itemsize

    # Comprobamos si el tamaño es consistente
    if bytes_totales % (num_canales * bytes_por_muestra) != 0:
        print("ADVERTENCIA: El tamaño del archivo no es un múltiplo perfecto del número de canales.")
        print("Esto podría indicar que el archivo está corrupto o se guardó incorrectamente.")
    
    print(f"Tamaño del archivo: {bytes_totales / 1e6:.2f} MB")

    # --- 2. Lectura y Reconstrucción de los Datos ---
    try:
        # Leemos el archivo binario como un único vector largo (1D)
        datos_planos = np.fromfile(filepath, dtype=dtype)
        
        # Calculamos el número de muestras (puntos en el tiempo)
        num_muestras = len(datos_planos) // num_canales
        
        # Reconstruimos la matriz a su forma original: (muestras, canales)
        # El -1 en reshape infiere la dimensión automáticamente.
        datos_reconstruidos = datos_planos.reshape((num_muestras, num_canales))

        print(f"¡Lectura exitosa!")
        print(f"Forma de los datos reconstruidos: {datos_reconstruidos.shape} (muestras, canales)")

    except Exception as e:
        print(f"ERROR: No se pudieron leer o reconstruir los datos. Causa: {e}")
        return

    # --- 3. Visualización ---
    print("Generando gráfico de una porción de los datos...")
    
    # Definimos cuántos segundos queremos visualizar
    segundos_a_plotear = 2
    muestras_a_plotear = int(segundos_a_plotear * freq_muestreo)
    
    # Nos aseguramos de no intentar plotear más muestras de las que existen
    if muestras_a_plotear > num_muestras:
        muestras_a_plotear = num_muestras

    # Creamos un vector de tiempo para el eje X
    tiempo = np.arange(muestras_a_plotear) / freq_muestreo

    # Seleccionamos algunos canales para no saturar el gráfico
    canales_a_plotear = [0, num_canales // 4, num_canales // 2]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, canal_idx in enumerate(canales_a_plotear):
        # Desplazamos cada canal verticalmente para que no se superpongan
        offset = i * 500 # Puedes ajustar este valor
        ax.plot(tiempo, datos_reconstruidos[:muestras_a_plotear, canal_idx] + offset, label=f'Canal {canal_idx}')

    ax.set_title(f"Verificación de Datos: Primeros {segundos_a_plotear} segundos")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (unidades de entero) + Offset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("Mostrando gráfico. Cierra la ventana del gráfico para terminar el script.")
    plt.show()


if __name__ == '__main__':
    # --- Cómo ejecutar este script ---
    # Uso: python verificar_dat.py <ruta_al_archivo.dat> <numero_de_canales>
    
    if len(sys.argv) != 3:
        print("Error en los argumentos.")
        print("Uso correcto: python verificar_dat.py ruta/a/tu/archivo.dat numero_de_canales")
        sys.exit(1)
        
    archivo_dat = sys.argv[1]
    try:
        canales = int(sys.argv[2])
    except ValueError:
        print("ERROR: El número de canales debe ser un entero.")
        sys.exit(1)

    verificar_dat(filepath=archivo_dat, num_canales=canales)
