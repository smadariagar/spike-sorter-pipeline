import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.widgets as sw
import probeinterface as pi

# =========================================================
# HELPER FUNCTIONS (Sin cambios)
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
# GENERAL AND SORTER PARAMETERS (EDIT ZONE - TRIDESCLOUS)
# =========================================================

# General settings
MEA_probe = True
sorter_name = 'tridesclous' 

# Sorter parameters dictionary para Tridesclous
sorter_params = {
    'detect_sign': -1,             # -1 para picos negativos (standard)
    'detect_threshold': 5.0,       # Umbral de detección basado en la desviación estándar del ruido (MAD)
    'common_ref_removal': False,   # Ponlo en True si tienes mucho ruido de red compartido (CMR)
    'n_jobs': -1                   # Usa todos los núcleos disponibles de tu procesador
}
# ==============================================================================
# TRIDESCLOUS: COMPLETE PARAMETER REFERENCE EN SPIKEINTERFACE
# ==============================================================================
# Tridesclous tiene muchos parámetros internos, pero SpikeInterface expone los 
# más críticos para facilitar el uso:
#
# 'filter' (bool): Si Tridesclous debe aplicar su propio filtro bandpass. 
#                  Se recomienda False si ya usas `spre.bandpass_filter`.
# 'freq_min' / 'freq_max': Límites del filtro interno (solo se usan si filter=True).
# 'detect_sign' (-1): Dirección del pico. -1 = negativo, 1 = positivo, 0 = ambos.
# 'detect_threshold' (5.0): Multiplicador del Median Absolute Deviation (MAD). 
#                           Un 5.0 significa que el pico debe ser 5 veces el nivel 
#                           de ruido basal. Baja a 4.0 o 4.5 si te pierdes picos pequeños.
# 'common_ref_removal' (False): Si es True, resta la mediana de todos los canales a 
#                               cada canal antes de la detección. Excelente para MEAs 
#                               si tienes ruido de 50Hz/60Hz o artefactos de movimiento.
# ==============================================================================

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # FILE SELECTION UI
    root = tk.Tk()
    root.withdraw()

    selected_file_paths = filedialog.askopenfilenames(
        title="Select recording files (You can select multiple)",
        filetypes=[("H5/RHS files", "*.h5 *.rhs"), ("H5 files", "*.h5"), ("RHS files", "*.rhs"), ("All files", "*.*")]
    )

    if not selected_file_paths:
        print("Operation canceled.")
        exit() 

    selected_file_paths = sorted(list(selected_file_paths))

    custom_name = simpledialog.askstring("Output Name", "Enter the name for this analysis session:")
    if not custom_name:
        print("No name provided. Operation canceled.")
        exit()

    input_folder = os.path.dirname(selected_file_paths[0])
    output_folder = os.path.join(input_folder, 'sorter_results/')
    os.makedirs(output_folder, exist_ok=True)

    sorting_output_folder = os.path.join(output_folder, f"sorting_{custom_name}")
    waveforms_folder = os.path.join(output_folder, f"waveforms_{custom_name}")

    # =========================================================
    # PHASE 1: DATA LOADING AND GEOMETRY
    # =========================================================
    recording_list = []

    if selected_file_paths[0].endswith('.h5'):
        print(f"Loading {len(selected_file_paths)} H5 file(s)...")
        for full_file_path in selected_file_paths:
            try:
                rec = se.read_mcsh5(full_file_path, stream_id='0')
                recording_list.append(rec)
            except IndexError:
                print(f"  -> WARNING: Skipping '{os.path.basename(full_file_path)}'")
            except Exception as e:
                print(f"  -> WARNING: Could not load '{os.path.basename(full_file_path)}'. Error: {e}")
     
    elif selected_file_paths[0].endswith('.rhs'):
        print(f"Loading {len(selected_file_paths)} RHS file(s)...")
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            rec = spre.unsigned_to_signed(rec)
            recording_list.append(rec)
    
    if not recording_list:
        print("\nError: No valid recordings were loaded. Operation canceled.")
        exit()

    # Combinar múltiples archivos
    if len(recording_list) > 1:
        recording = sc.concatenate_recordings(recording_list)
    else:
        recording = recording_list[0]

    num_channels = recording.get_num_channels()
    print(f"Loaded channels: {num_channels}")

    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'
    probe = create_probe(is_mea=MEA_probe, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    print(f"Effective channels for sorting: {recording.get_num_channels()}")

    # =========================================================
    # PHASE 2: SPIKE SORTING
    # =========================================================
    print("Applying chained preprocessing...")
    # El filtro se mantiene aquí, por eso le decimos a TDC que NO filtre internamente
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    print(f"\nStarting {sorter_name}...")
    
    sorting_result = ss.run_sorter(
        sorter_name=sorter_name,
        recording=recording,
        folder=sorting_output_folder,  
        remove_existing_folder=True,
        **sorter_params
    )

    found_units = sorting_result.get_unit_ids()
    print("\n=== Sorting Finished ===")
    print(f"Potential neurons (clusters) found: {len(found_units)}")
    print(f"Unit IDs: {found_units}")

    # =========================================================
    # PHASE 3: WAVEFORM EXTRACTION
    # =========================================================
    print("\nCreating the waveform analyzer...")
    analyzer = sc.create_sorting_analyzer(
        sorting=sorting_result,
        recording=recording,
        format="binary_folder",
        folder=waveforms_folder,
        overwrite=True
    )

    print("Computing spikes and extracting waveforms...")
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)                   
    analyzer.compute("templates")                                                
    print("Waveforms successfully extracted!")