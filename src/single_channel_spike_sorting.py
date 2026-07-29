import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import probeinterface as pi

# =========================================================
# PROBE 
# =========================================================
def create_probe(is_mea, file_type, num_channels, pitch=200, radius=15):
    """
    Genera el mapa de electrodos. Se mantiene igual a tu versión original.
    """
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
# GENERAL AND SORTER PARAMETERS 
# =========================================================
MEA_probe = True

# USAMOS MOUNTAINSORT5 (Mejor para single-channel sorting)
sorter_name = 'mountainsort5'   
sorter_params = {
    'detect_threshold': 4.0,    # 4.0                            
    'detect_sign': -1,           # -1                          
    'filter': False,    
    'whiten': False,            # True                 
}

# # USAMOS TRIDESCLOUS
# sorter_name = 'tridesclous'   
# sorter_params = {
#     'detect_sign': -1,            # -1 para espigas negativas, 1 para positivas, 0 para ambas
#     'radius_um': 0.0,        
#     'detect_threshold': 2.5 , #4.0,      # Umbral de detección (ajusta según el nivel de ruido de tu MEA)
#     'freq_min': 300.0,            # Tridesclous maneja su propio filtrado internamente
#     'freq_max': 6000.0,
#     'common_ref_removal': False   # Desactivado porque estás aislando y procesando un solo canal a la vez
# }

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # 1. FILE SELECTION UI
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
    output_folder = os.path.join(input_folder, f'single_channel_sorting/{custom_name}/')
    os.makedirs(output_folder, exist_ok=True)

    # 2. DATA LOADING AND GEOMETRY
    recording_list = []

    if selected_file_paths[0].endswith('.h5'):
        print(f"Loading {len(selected_file_paths)} H5 file(s)...")
        for full_file_path in selected_file_paths:
            try:
                rec = se.read_mcsh5(full_file_path, stream_id='0')
                recording_list.append(rec)
            except Exception as e:
                print(f"  -> WARNING: Could not load '{os.path.basename(full_file_path)}'. Error: {e}")
     
    elif selected_file_paths[0].endswith('.rhs'):
        print(f"Loading {len(selected_file_paths)} RHS file(s)...")
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            rec = spre.unsigned_to_signed(rec)
            recording_list.append(rec)
    
    if not recording_list:
        print("\nError: No valid recordings were loaded.")
        exit()

    recording = sc.concatenate_recordings(recording_list) if len(recording_list) > 1 else recording_list[0]
    num_channels = recording.get_num_channels()
    
    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'
    probe = create_probe(is_mea=MEA_probe, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    # 3. PREPROCESSING & CACHING (ENTIRE DATASET)
    print("Applying chained preprocessing...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    cached_folder = os.path.join(output_folder, "cached_binary_full")
    print(f"Saving preprocessed full data to disk...")
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)
    recording_saved = recording.save(folder=cached_folder, format='binary', overwrite=True, **job_kwargs)
    
    fs = recording_saved.get_sampling_frequency()

    # =========================================================
    # 4. SINGLE CHANNEL SORTING LOOP
    # =========================================================
    all_spikes_data = []
    
    channel_ids = recording_saved.get_channel_ids()
    print(f"\n=== Starting Single-Channel Sorting for {len(channel_ids)} channels ===")

    for chan_id in channel_ids:
        print(f"\n---> Processing Channel ID: {chan_id}")
        
        # Aislar solo este canal
        rec_single_chan = recording_saved.select_channels(channel_ids=[chan_id])
        
        chan_output_folder = os.path.join(output_folder, f"sorting_ch_{chan_id}")
        
        try:
            # Correr el sorter solo en este canal
            sorting_result = ss.run_sorter(
                sorter_name=sorter_name,
                recording=rec_single_chan, 
                folder=chan_output_folder,  
                remove_existing_folder=True,
                **sorter_params  
            )
            
            found_units = sorting_result.get_unit_ids()
            print(f"     Found {len(found_units)} units in channel {chan_id}")
            
            # Extraer tiempos de espigas y guardarlos en nuestra lista maestra
            for unit_idx, unit_id in enumerate(found_units):
                # Generamos un ID único global, ej: "Ch47_U0"
                global_unit_id = f"Ch{chan_id}_U{unit_idx}"
                
                spike_frames = sorting_result.get_unit_spike_train(unit_id)
                
                for frame in spike_frames:
                    all_spikes_data.append({
                        'Electrode_ID': chan_id,
                        'Neuron_ID': global_unit_id,
                        'Spike_Frame': frame,
                        'Spike_Time_Seconds': frame / fs
                    })
                    
        except Exception as e:
            print(f"     [ERROR] Sorter failed on channel {chan_id}. Skipping. Error: {e}")

    # =========================================================
    # 5. EXPORT FINAL CONSOLIDATED DATA
    # =========================================================
    if len(all_spikes_data) > 0:
        print("\n=== All channels processed. Saving consolidated data ===")
        df_spikes = pd.DataFrame(all_spikes_data)
        
        # Ordenar por tiempo de ocurrencia
        df_spikes = df_spikes.sort_values(by='Spike_Time_Seconds').reset_index(drop=True)
        
        csv_path = os.path.join(output_folder, f"all_spikes_consolidated_{custom_name}.csv")
        df_spikes.to_csv(csv_path, index=False)
        
        print(f"Success! Total spikes found across all channels: {len(df_spikes)}")
        print(f"Data saved to: {csv_path}")
    else:
        print("\nNo spikes were found in any channel.")