import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

# =========================================================
# CONFIGURACIÓN
# =========================================================
# Frecuencia de muestreo (Sample Rate) de tu equipo en Hz. 
# Modifica esto si tu equipo grabó a 20kHz, 25kHz, etc.
DEFAULT_FS = 40000 

def fix_electrode_id(raw_id):
    """
    Toma el ID original del TXT y lo desglosa.
    Ej: 8401 -> Electrodo 84, Neurona 1.
    Luego corrige el error de suma (+64 en vez de +32) del profesor.
    """
    # Los últimos dos dígitos son la neurona (ej. 01, 02), el resto es el electrodo
    electrode_raw = raw_id // 100
    unit_idx = raw_id % 100
    
    # Corrección matemática del bloque superior
    if electrode_raw >= 65:
        electrode_real = electrode_raw - 32
    else:
        electrode_real = electrode_raw
        
    return electrode_real, unit_idx

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # 1. INTERFAZ DE SELECCIÓN DE ARCHIVOS
    root = tk.Tk()
    root.withdraw()

    print("Por favor, selecciona los archivos TXT (MEA36 A.txt, B.txt, C.txt)...")
    selected_files = filedialog.askopenfilenames(
        title="Selecciona los archivos TXT del profesor (Puedes seleccionar varios)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if not selected_files:
        print("Operación cancelada.")
        exit()

    # Preguntar por la frecuencia de muestreo para calcular el tiempo en segundos
    fs_input = simpledialog.askinteger(
        "Frecuencia de Muestreo", 
        "Ingresa la frecuencia de muestreo de la grabación en Hz (ej. 40000):",
        initialvalue=DEFAULT_FS
    )
    
    if not fs_input:
        print("No se ingresó frecuencia. Operación cancelada.")
        exit()
        
    fs = fs_input

    # 2. PROCESAMIENTO Y CORRECCIÓN DE DATOS
    all_spikes = []
    
    print(f"\nProcesando {len(selected_files)} archivos a {fs} Hz...")

    for file_path in selected_files:
        # Extraer una etiqueta del archivo (ej. de "MEA36 A.txt" extrae "A")
        file_label = os.path.basename(file_path).replace('.txt', '').split(' ')[-1]
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                raw_id = int(parts[0])
                frame = int(parts[1])
                
                # Aplicar la corrección
                electrode_real, unit_idx = fix_electrode_id(raw_id)
                
                # Crear el ID global de la neurona (ej. "Ch47_U0")
                neuron_id = f"Ch{electrode_real}_U{unit_idx}"
                time_seconds = frame / fs
                
                all_spikes.append({
                    'Electrode_ID': electrode_real,
                    'Neuron_ID': neuron_id,
                    'Spike_Frame': frame,
                    'Spike_Time_Seconds': time_seconds,
                    'Clasificacion_Origen': file_label
                })

    # 3. EXPORTAR A CSV CONSOLIDADO
    if len(all_spikes) > 0:
        df_spikes = pd.DataFrame(all_spikes)
        
        # Ordenar cronológicamente por el frame
        print("Ordenando eventos cronológicamente...")
        df_spikes = df_spikes.sort_values(by='Spike_Frame').reset_index(drop=True)
        
        # Generar ruta de guardado
        output_dir = os.path.dirname(selected_files[0])
        output_csv = os.path.join(output_dir, "MEA36_all_spikes_corrected.csv")
        
        df_spikes.to_csv(output_csv, index=False)
        
        print(f"\n¡Éxito! Se procesaron y corrigieron {len(df_spikes)} espigas en total.")
        print(f"Archivo guardado en: {output_csv}")
        
        # Pequeño resumen de calidad
        print("\n--- Resumen de Datos ---")
        print(df_spikes['Clasificacion_Origen'].value_counts().to_string())
    else:
        print("No se encontraron datos válidos en los archivos seleccionados.")