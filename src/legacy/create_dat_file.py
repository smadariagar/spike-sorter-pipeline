import os
import gc
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog

from utils.load_intan_rhs_format import read_data
#from utils.Butter_bandpass_filter import Butter_bandpass_filter

# Configuración y diálogo selección carpetas
root = tk.Tk()
root.withdraw()

# solo muestra los archivos
file_types = [
    ("Intan RHS files", "*.rhs"),
    ("All files", "*.*")
]

# Usa 'askopenfilename' para mostrar los archivos
ruta_archivo_seleccionado = filedialog.askopenfilename(
    title="Select ANY .rhs file within the folder",
    filetypes=file_types
)

# if cancel
if not ruta_archivo_seleccionado:
    print("No files were selected. Operation canceled")
    exit() 

carpeta_entrada = os.path.dirname(ruta_archivo_seleccionado)
print(f"Selected input folder: {carpeta_entrada}")

# Path carpeta seleccionada
carpeta_salida = os.path.join(carpeta_entrada, "outputs")

# Crea la carpeta output dentro de la carpeta seleccionada
os.makedirs(carpeta_salida, exist_ok=True)
print(f"Output files will be saved in: {carpeta_salida}")

############

# Busca los archivos .rhs en la carpeta de entrada
patron_busqueda = os.path.join(carpeta_entrada, "*.rhs")
lista_archivos_rhs = glob.glob(patron_busqueda)

if not lista_archivos_rhs:
    print(f"No .rhs files were found in {carpeta_entrada}")
    exit()

carpeta_entrada = os.path.dirname(ruta_archivo_seleccionado)
print(f"{len(lista_archivos_rhs)} .rhs files were found to be processed...")

for ind, ruta_archivo_completa in enumerate(lista_archivos_rhs):

    nombre_base = os.path.basename(ruta_archivo_completa)
    nombre_sin_extension = os.path.splitext(nombre_base)[0]

    print(f"\n--- Procesing: {nombre_base} ---")

    # nombre completo del archivo y lectura de los datos
    rhsfile = read_data(ruta_archivo_completa)

    # analógico del dato tipo 'amplifier_data'
    analogico = rhsfile['amplifier_data']
    del rhsfile
    
    # separación de los 64 canales en dos grupos de 32
    bloque_1_a_32 = analogico[:32]  # Canales del 0 al 31
    bloque_33_a_64 = analogico[32:] # Canales del 32 al 63

    # el uso de esta constante está del código de la cata 
    # y no se si debe ir o no pero lo dejé
    constante = 10000000

    # ----- Generación del archivo con los primero 32 canales
    raw_para_guardar = np.ascontiguousarray((bloque_1_a_32 * 1e-6 * constante).T)
    raw_para_guardar = raw_para_guardar.astype(np.int16)
    del bloque_1_a_32
    
    # nombre salida primer archivo (primero 32 canales)
    nombre_archivo_1 = f"{nombre_sin_extension}_chan_00_31.{ind:03}"    
    output_path = os.path.join(carpeta_salida, nombre_archivo_1)

    # se guarda el archivo
    print(f"Saving: {nombre_archivo_1}")
    with open(output_path, 'wb') as f:
        f.write(raw_para_guardar.tobytes()) # .tobytes() es la forma correcta de escribir un array numpy
    del raw_para_guardar

    # ----- Generación del archivo con los segudos 32 canales
    raw_para_guardar = np.ascontiguousarray((bloque_33_a_64 * 1e-6 * constante).T)
    raw_para_guardar = raw_para_guardar.astype(np.int16)
    del bloque_33_a_64
    
    # se crea el nombre del archivo
    nombre_archivo_2 = f"{nombre_sin_extension}_chan_32_63.{ind:03}"    
    output_path = os.path.join(carpeta_salida, nombre_archivo_2)

    # se guarda el archivo
    print(f"Saving: {nombre_archivo_2}")
    with open(output_path, 'wb') as f:
        f.write(raw_para_guardar.tobytes()) # .tobytes() es la forma correcta de escribir un array numpy
    del raw_para_guardar

    gc.collect()