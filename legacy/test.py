import os
import gc
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog
import h5py

def process_and_save_binary_padded(h5_dataset, channel_start, channel_end, output_path, chunk_size=200000):
    """
    Lee un rango de canales y guarda en binario.
    Si faltan canales en el H5, rellena con CEROS para completar el bloque requerido.
    """
    total_samples = h5_dataset.shape[1]
    total_h5_channels = h5_dataset.shape[0]
    
    # Cuántos canales necesitamos generar en el archivo final
    channels_needed = channel_end - channel_start
 
    # Si pedimos hasta el 64, pero hay 60, el limite real será 60.
    real_read_limit = min(channel_end, total_h5_channels)
    
    # Verificamos si habrá que rellenar
    needs_padding = real_read_limit < channel_end
    padding_count = channel_end - real_read_limit
    
    FACTOR_CONVERSION = 10 
    
    print(f"  -> Guardando en: {os.path.basename(output_path)}")
    print(f"     Rango solicitado: {channel_start} a {channel_end-1}")
    if needs_padding:
        print(f"     [AVISO] Se leerán {real_read_limit - channel_start} canales reales y se rellenarán {padding_count} canales con ceros.")
    
    with open(output_path, 'wb') as f_out:
        
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            current_chunk_width = end - start
            
            # Si el inicio solicitado ya es mayor que los canales que existen, devolvemos vacío
            if channel_start < total_h5_channels:
                data_chunk = h5_dataset[channel_start:real_read_limit, start:end]
            else:
                data_chunk = np.empty((0, current_chunk_width))

            # Rellena con ceros
            if needs_padding:
                # Crear matriz de ceros (Canales faltantes x Muestras actuales)
                zeros_padding = np.zeros((padding_count, current_chunk_width), dtype=data_chunk.dtype)
                
                if data_chunk.size > 0:
                    # Unir datos reales + ceros (Stack vertical)
                    data_chunk = np.vstack([data_chunk, zeros_padding])
                else:
                    # Si no había datos reales (ej. pedimos canales 70-80 y hay 60), todo es ceros
                    data_chunk = zeros_padding
            
            data_chunk = data_chunk * FACTOR_CONVERSION
            
            # Clip y Conversión
            np.clip(data_chunk, -32767, 32767, out=data_chunk)
            data_int16 = data_chunk.astype(np.int16)
            
            # Transponer (Samples x Canales)
            data_to_save = np.ascontiguousarray(data_int16.T)
            
            f_out.write(data_to_save.tobytes())
            
            if start % (chunk_size * 5) == 0:
                progreso = (end / total_samples) * 100
                print(f"     Progreso: {progreso:.1f}%", end='\r')
            
            del data_chunk, data_int16, data_to_save
            
    print(f"     Progreso: 100.0% - Completado.")
    gc.collect()

##########################

root = tk.Tk()
root.withdraw()

ruta_archivo_seleccionado = filedialog.askopenfilename(
    title="Selecciona un archivo .h5",
    filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
)

if not ruta_archivo_seleccionado:
    exit() 

carpeta_entrada = os.path.dirname(ruta_archivo_seleccionado)
carpeta_salida = os.path.join(carpeta_entrada, "outputs_binary")
os.makedirs(carpeta_salida, exist_ok=True)

patron_busqueda = os.path.join(carpeta_entrada, "*.h5")
lista_archivos_h5 = glob.glob(patron_busqueda)

print(f"Procesando {len(lista_archivos_h5)} archivos...")

for ind, ruta_archivo_completa in enumerate(lista_archivos_h5):
    nombre_base = os.path.basename(ruta_archivo_completa)
    nombre_sin_extension = os.path.splitext(nombre_base)[0]
    
    print(f"\n{'='*60}")
    print(f"Procesando: {nombre_base}")
    
    with h5py.File(ruta_archivo_completa, 'r') as f:
        try:
            dset = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']
            print(f"  Shape H5: {dset.shape}")
            
            # --- ARCHIVO 1: Canales 0 a 31 ---
            # Este siempre debería existir completo, pero usamos la función segura igual
            nombre_archivo_1 = f"{nombre_sin_extension}_chan_00_31.{ind:03}"
            path_out_1 = os.path.join(carpeta_salida, nombre_archivo_1)
            
            process_and_save_binary_padded(
                dset, 
                channel_start=0, 
                channel_end=32, # Python excluye el final, así que esto es 0...31
                output_path=path_out_1
            )

            # --- ARCHIVO 2: Canales 32 a 63 ---
            # AQUÍ ESTÁ EL CAMBIO: Forzamos la creación aunque haya menos canales
            nombre_archivo_2 = f"{nombre_sin_extension}_chan_32_63.{ind:03}"
            path_out_2 = os.path.join(carpeta_salida, nombre_archivo_2)
            
            process_and_save_binary_padded(
                dset, 
                channel_start=32, 
                channel_end=64, # Python excluye el final, así que esto es 32...63
                output_path=path_out_2
            )
                
        except KeyError as e:
            print(f"  [ERROR] Estructura H5 incorrecta: {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")

print(f"\n{'='*60}\nListo.")