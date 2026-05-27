import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.core as sc
import probeinterface as pi

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def create_probe(is_mea, file_type, num_channels, pitch=200, radius=15):
    if not is_mea:
        print("Assigning linear spatial electrode map...")
        linear_probe = pi.generate_linear_probe(num_elec=num_channels, ypitch=20) 
        linear_probe.set_device_channel_indices(np.arange(num_channels))
        return linear_probe

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mea_mapping.json')
    with open(json_path, 'r') as f:
        mea_mapping = json.load(f)
    
    map_key = "channel_mapping_rhs" if file_type == 'rhs' else "channel_mapping_h5"
    list_2_map = mea_mapping[map_key]

    print(f"Assigning MEA spatial map for {num_channels} channels...")
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

    if len(valid_channel_indices) == 0:
        raise ValueError("Error: No valid channels found to map.")

    probe_mea.set_contacts(
        positions=np.array(positions), 
        shapes='circle', 
        shape_params={'radius': radius}
    )

    probe_mea.set_device_channel_indices(valid_channel_indices)
    return probe_mea

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # 1. INTERFAZ DE SELECCIÓN
    root = tk.Tk()
    root.withdraw()

    selected_file_paths = filedialog.askopenfilenames(
        title="Select recording files to PREPROCESS and EXPORT",
        filetypes=[("H5/RHS files", "*.h5 *.rhs"), ("H5 files", "*.h5"), ("RHS files", "*.rhs")]
    )

    if not selected_file_paths:
        print("Operation canceled.")
        exit() 

    selected_file_paths = sorted(list(selected_file_paths))

    custom_name = simpledialog.askstring("Output Name", "Enter the name for this session:")
    if not custom_name:
        print("No name provided. Operation canceled.")
        exit()

    input_folder = os.path.dirname(selected_file_paths[0])
    
    # 2. CARGA Y CONCATENACIÓN
    recording_list = []
    if selected_file_paths[0].endswith('.h5'):
        print(f"Loading {len(selected_file_paths)} H5 file(s)...")
        for full_file_path in selected_file_paths:
            try:
                rec = se.read_mcsh5(full_file_path, stream_id='0')
                recording_list.append(rec)
            except Exception as e:
                print(f"  -> WARNING: Skipping '{os.path.basename(full_file_path)}'. Error: {e}")
     
    elif selected_file_paths[0].endswith('.rhs'):
        print(f"Loading {len(selected_file_paths)} RHS file(s)...")
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            rec = spre.unsigned_to_signed(rec)
            recording_list.append(rec)
    
    if not recording_list:
        print("\nError: No valid recordings were loaded. Operation canceled.")
        exit()

    recording = sc.concatenate_recordings(recording_list) if len(recording_list) > 1 else recording_list[0]
    num_channels = recording.get_num_channels()

    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'
    probe = create_probe(is_mea=True, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    # 3. PREPROCESAMIENTO (Limpieza de ruido)
    print("\nApplying bandpass filter (300-6000 Hz)...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    # 4. EXPORTACIÓN PARA TRIDESCLOUS NATIVO
    fs = recording.get_sampling_frequency()
    export_folder = os.path.join(input_folder, f"tdc_ready_{custom_name}")
    print(f"\nExporting clean data to: {export_folder}")

    # Guarda el registro en formato binario (.raw)
    recording_saved = recording.save(folder=export_folder, format='binary', n_jobs=-1, overwrite=True)
    
    # Guarda el mapa de electrodos en formato .prb (nativo de Tridesclous)
    probegroup = pi.ProbeGroup()
    probegroup.add_probe(probe)
    
    prb_path = os.path.join(export_folder, "mea_probe.prb")
    pi.write_prb(prb_path, probegroup)

    print("\n" + "="*50)
    print("¡EXPORTACIÓN FINALIZADA CON ÉXITO!")
    print("="*50)
    print("Tus datos limpios y mapeados están listos para la GUI de Tridesclous.")
    print("\nCUANDO ABRAS 'tdc' Y CREES UN NUEVO DATASET, USA ESTOS PARÁMETROS:")
    print(f"Archivo de datos: {export_folder}/traces_cached_seg0.raw")
    print(f"Frecuencia de muestreo (Sample Rate): {fs}")
    print(f"Número de canales (Num Channels): {recording.get_num_channels()}")
    print(f"Formato de datos (dtype): float32 (¡Importante! El filtro lo cambió a float32)")
    print(f"Archivo de geometría (PRB): {prb_path}")
    print("="*50)