import pandas as pd
import numpy as np
import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.core as sc
import spikeinterface_gui as sig
import probeinterface as pi

# =========================================================
# 0. FUNCION DEL MAPA DE ELECTRODOS (Para reconstruir la data en RAM)
# =========================================================
def create_probe(is_mea, file_type, num_channels, pitch=200, radius=15):
    if not is_mea:
        linear_probe = pi.generate_linear_probe(num_elec=num_channels, ypitch=20) 
        linear_probe.set_device_channel_indices(np.arange(num_channels))
        return linear_probe

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mea_mapping.json')
    with open(json_path, 'r') as f:
        mea_mapping = json.load(f)
    
    map_key = "channel_mapping_rhs" if file_type == 'rhs' else "channel_mapping_h5"
    list_2_map = mea_mapping[map_key]

    probe_mea = pi.Probe(ndim=2, si_units='um')
    positions, valid_channel_indices = [], []
    
    for i, num in enumerate(list_2_map):
        num_str = str(num)
        if num_str == '0':
            continue
        x = (int(num_str[0]) - 1) * pitch
        y = (8 - int(num_str[1])) * pitch
        positions.append([x, y])
        valid_channel_indices.append(i)

    probe_mea.set_contacts(positions=np.array(positions), shapes='circle', shape_params={'radius': radius})
    probe_mea.set_device_channel_indices(valid_channel_indices)
    return probe_mea

# =========================================================
# 1. SELECCIÓN DE ARCHIVOS Y LECTURA DEL HEADER
# =========================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    # A. Seleccionar SOLAMENTE el archivo de resumen (Header)
    summary_path = filedialog.askopenfilename(
        title="1. Selecciona el archivo de resumen (analysis_summary_... .txt)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not summary_path:
        print("Operación cancelada.")
        exit()

    # B. Parsear el archivo txt para encontrar las rutas originales
    selected_file_paths = []
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Buscamos la línea que guardamos con el formato "(Path: ruta_del_archivo)"
            if line.startswith("(Path:") and line.endswith(")"):
                # Limpiamos el string para quedarnos solo con la ruta pura
                file_path = line.replace("(Path:", "").rstrip(")").strip()
                
                # Verificamos que el archivo original no haya sido movido o borrado
                if os.path.exists(file_path):
                    selected_file_paths.append(file_path)
                else:
                    print(f"[ADVERTENCIA] Archivo original no encontrado en la ruta: {file_path}")

    if not selected_file_paths:
        print("\n[ERROR] No se encontraron rutas válidas de archivos originales en el resumen.")
        print("Asegúrate de no haber borrado o movido los archivos .h5 originales.")
        exit()
        
    print(f"\n[+] Se leyeron {len(selected_file_paths)} archivos originales desde el resumen.")

    # C. Buscar automáticamente el CSV en la misma carpeta que el txt
    folder_path = os.path.dirname(summary_path)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith("all_spikes") and f.endswith(".csv")]
    
    if not csv_files:
        print(f"\n[ERROR] No se encontró el archivo CSV en la carpeta: {folder_path}")
        exit()
        
    csv_path = os.path.join(folder_path, csv_files[0])
    print(f"[+] CSV encontrado automáticamente: {os.path.basename(csv_path)}")

    # =========================================================
    # 2. RECONSTRUCCIÓN VIRTUAL "AL VUELO" (CERO USO DE DISCO)
    # =========================================================
    print("\nReconstruyendo grabación desde los originales...")
    recording_list = []
    if selected_file_paths[0].endswith('.h5'):
        for full_file_path in selected_file_paths:
            try:
                recording_list.append(se.read_mcsh5(full_file_path, stream_id='0'))
            except Exception as e:
                pass 
                
    recording = sc.concatenate_recordings(recording_list) if len(recording_list) > 1 else recording_list[0]
    num_channels = recording.get_num_channels()
    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'
    
    probe = create_probe(is_mea=True, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)
    
    # Aplicamos el filtro en la memoria RAM
    print("Aplicando filtro bandpass al vuelo...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    fs = recording.get_sampling_frequency()

    # =========================================================
    # 3. CARGAR RESULTADOS DESDE EL CSV
    # =========================================================
    print(f"\nCargando tiempos de espigas desde el CSV...")
    df = pd.read_csv(csv_path)

    unit_dict = {}
    for unit_id, group in df.groupby("Neuron_ID"):
        unit_dict[unit_id] = group["Spike_Frame"].to_numpy(dtype=int)

    print(f"Total de unidades aisladas: {len(unit_dict)}")
    unified_sorting = sc.NumpySorting.from_unit_dict([unit_dict], sampling_frequency=fs)

    # =========================================================
    # 4. EXTRACCIÓN Y VISUALIZACIÓN
    # =========================================================
    analyzer_folder = os.path.join(folder_path, "unified_analyzer")

    if os.path.exists(analyzer_folder):
        shutil.rmtree(analyzer_folder)

    print("\nCreando Analizador Global...")
    analyzer = sc.create_sorting_analyzer(
        sorting=unified_sorting,
        recording=recording,
        format="memory", # <--- IMPORTANTE: Todo se calcula en RAM
        folder=None      # <--- IMPORTANTE: No crea carpeta pesada
    )

    print("Extrayendo formas de onda (Esto tomará un par de minutos, ten paciencia)...")
    job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates")
    analyzer.compute("noise_levels")

    print("\n¡Abriendo Interfaz Gráfica!")
    app = sig.run_mainwindow(analyzer)