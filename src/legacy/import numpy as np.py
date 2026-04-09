import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def inspect_binary():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Selecciona el archivo binario (.000)",
        filetypes=[("Binary files", "*.000"), ("All files", "*.*")]
    )

    if not file_path:
        print("Operación cancelada.")
        return

    # Parámetros esperados (Ajusta según tu caso)
    NUM_CHANNELS = 32 
    DTYPE = np.int16 # El estándar de 16 bits que usas

    file_size = os.path.getsize(file_path)
    
    print(f"\n{'='*40}")
    print(f"ARCHIVO: {os.path.basename(file_path)}")
    print(f"Tamaño total: {file_size} bytes")
    
    try:
        # Leemos solo el encabezado para no saturar la memoria
        # Leemos 10 muestras completas (10 muestras * 32 canales)
        num_items_to_read = 10 * NUM_CHANNELS
        
        data = np.fromfile(file_path, dtype=DTYPE, count=num_items_to_read)
        
        print(f"Tipo de dato detectado (Numpy): {data.dtype}")
        print(f"Tamaño de cada valor: {data.itemsize * 8} bits ({data.itemsize} bytes)")
        print(f"Cantidad de valores leídos: {len(data)}")
        print(f"{'='*40}")
        print("Primeros 10 valores (en orden de lectura):")
        
        for i, val in enumerate(data[:10]):
            print(f" Valor {i}: {val}")
            
        # Intentar detectar si hay ceros constantes o valores fuera de rango
        if np.all(data == 0):
            print("\n[ALERTA] Los primeros valores son todos CERO. Revisa el escalamiento.")
            
    except Exception as e:
        print(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    inspect_binary()