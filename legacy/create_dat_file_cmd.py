import os
import gc
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog

import mcpy
import h5py

# Configuración y diálogo selección carpetas
root = tk.Tk()
root.withdraw()

# solo muestra los archivos
file_types = [
    ("MCS files", "*.mcd"),
    ("All files", "*.*")
]

# Usa 'askopenfilename' para mostrar los archivos
ruta_archivo_seleccionado = filedialog.askopenfilename(
    title="Select ANY .mcd file within the folder",
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
patron_busqueda = os.path.join(carpeta_entrada, "*.mcd")
lista_archivos_mcd = glob.glob(patron_busqueda)

if not lista_archivos_mcd:
    print(f"No .mcd files were found in {carpeta_entrada}")
    exit()

carpeta_entrada = os.path.dirname(ruta_archivo_seleccionado)
print(f"{len(lista_archivos_mcd)} .mcd files were found to be processed...")

for ind, ruta_archivo_completa in enumerate(lista_archivos_mcd):

    nombre_base = os.path.basename(ruta_archivo_completa)
    nombre_sin_extension = os.path.splitext(nombre_base)[0]

    print(f"\n--- Procesing: {nombre_base} ---")

    try:
        print(f"  Reading {nombre_base} with neo.io.McsIO...")
        # Usar neo.io.McsIO para leer el archivo .mcd
        reader = McsIO(filename=ruta_archivo_completa)
        block = reader.read_block()

        # Asumimos que los datos están en el primer segmento
        if not block.segments:
            print("  Warning: No 'segments' found. Skipping file.")
            continue
        seg = block.segments[0]

        # Asumimos que los datos analógicos son el primer 'analogsignal'
        if not seg.analogsignals:
            print("  Warning: No 'analogsignals' found. Skipping file.")
            continue
        
        # Cargar la señal analógica. 
        # neo la carga como (samples, channels)
        # La transponemos a (channels, samples) para que coincida con tu script original
        analogico = seg.analogsignals[0].load().magnitude.T
        print(f"  Data shape loaded (channels, samples): {analogico.shape}")

    except Exception as e:
        print(f"  ERROR: Could not read {nombre_base} with neo: {e}")
        print("  Skipping file.")
        continue

    try:
        # separación de los 64 canales en dos grupos de 32 (MISMA LÓGICA QUE TENÍAS)
        bloque_1_a_32 = analogico[:32]  # Canales del 0 al 31
        bloque_33_a_64 = analogico[32:] # Canales del 32 al final (como en tu script original)
        
        # Si quisieras *exactamente* 32 canales (32 al 63), usarías:
        # bloque_33_a_64 = analogico[32:64] 
        
        if bloque_1_a_32.shape[0] != 32:
             print(f"  Warning: Block 1 does not have 32 channels. Shape is {bloque_1_a_32.shape}")
        if bloque_33_a_64.shape[0] == 0:
             print(f"  Warning: Block 2 has 0 channels. Check channel count (total: {analogico.shape[0]})")


    except IndexError as e:
        print(f"  ERROR: Failed to split channels. Does the file have 64 channels? Error: {e}")
        continue
    except Exception as e:
        print(f"  ERROR: Unexpected error splitting channels: {e}")
        continue

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