import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import probeinterface as pi
import shutil  
import gc      

# =========================================================
# PROBE 
# =========================================================
def create_probe(is_mea, file_type, num_channels, pitch=200, radius=15):
    """
    Generates the electrode map depending on the geometry and number of channels.
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
sorter_name = 'mountainsort5'   

sorter_params = {
    # 1. Sensitivity Adjustment (Global Noise)
    'detect_threshold': 4.0,        
    'detect_sign': -1,              
    
    # 2. Overclustering Adjustment (Physiological Drift)
    'npca_per_channel': 8,          
    'snippet_T1': 30,               
    'snippet_T2': 30,               
    
    # 3. Training Adjustment (Temporal dilution)
    'scheme2_training_duration_sec': 900,               
    'scheme2_max_num_snippets_per_training_batch': 500, 
    'scheme2_training_recording_sampling_mode': 'uniform', 
    
    # 4. General Parameters
    'filter': False,                
    'whiten': True,                 
    'n_jobs': -1                    
}

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # 1. FILE SELECTION UI
    root = tk.Tk()
    root.withdraw() # Hides the main empty window

    selected_file_paths = filedialog.askopenfilenames(
        title="Select recording files (You can select multiple)",
        filetypes=[("H5/RHS files", "*.h5 *.rhs"), ("H5 files", "*.h5"), ("RHS files", "*.rhs"), ("All files", "*.*")]
    )

    if not selected_file_paths:
        print("Operation canceled.")
        root.destroy()
        exit() 

    selected_file_paths = sorted(list(selected_file_paths))

    custom_name = simpledialog.askstring("Output Name", "Enter the name for this analysis session:")
    if not custom_name:
        print("No name provided. Operation canceled.")
        root.destroy()
        exit()

    # Ask the user if they want to keep the heavy binary cache
    keep_cached_binary = messagebox.askyesno(
        "Keep Binary Cache", 
        "Do you want to keep the heavy 'cached_binary_full' folder?\n\n"
        "YES: Keep it only if you will use Phy for manual curation (It takes several GBs).\n"
        "NO: Delete it at the end to save hard drive space."
    )

    # ---> CRITICAL FIX: Destroy the UI completely right here so it doesn't freeze on the screen
    root.destroy()

    input_folder = os.path.dirname(selected_file_paths[0])
    output_folder = os.path.join(input_folder, f'single_channel_sorting/{custom_name}/')
    os.makedirs(output_folder, exist_ok=True)

    # 1.5 GENERATE SUMMARY FILE (HEADER)
    summary_txt_path = os.path.join(output_folder, f"analysis_summary_{custom_name}.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("              SPIKE SORTING ANALYSIS SUMMARY             \n")
        f.write("=========================================================\n\n")
        f.write(f"Session Name: {custom_name}\n")
        f.write(f"Kept Binary Cache: {keep_cached_binary}\n\n") 
        
        f.write("--- FILES USED ---\n")
        for file_path in selected_file_paths:
            f.write(f" * {os.path.basename(file_path)}\n")
            f.write(f"   (Path: {file_path})\n")
        
        f.write("\n--- SORTER CONFIGURATION ---\n")
        f.write(f"Algorithm: {sorter_name}\n")
        f.write("Parameters:\n")
        for key, value in sorter_params.items():
            f.write(f" * {key}: {value}\n")
    print(f"Summary file created at: {summary_txt_path}")

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

    # Temporarily saved to disk for efficient sorting
    cached_folder = os.path.join(output_folder, "cached_binary_full")
    print(f"Saving preprocessed full data to disk...")
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)
    recording_saved = recording.save(folder=cached_folder, format='binary', overwrite=True, **job_kwargs)
    
    fs = recording_saved.get_sampling_frequency()

    # 4. SINGLE CHANNEL SORTING LOOP
    all_spikes_data = []
    
    channel_ids = recording_saved.get_channel_ids()
    print(f"\n=== Starting Single-Channel Sorting for {len(channel_ids)} channels ===")

    for chan_id in channel_ids:
        print(f"\n---> Processing Channel ID: {chan_id}")
        
        rec_single_chan = recording_saved.select_channels(channel_ids=[chan_id])
        chan_output_folder = os.path.join(output_folder, f"sorting_ch_{chan_id}")
        
        try:
            sorting_result = ss.run_sorter(
                sorter_name=sorter_name,
                recording=rec_single_chan, 
                folder=chan_output_folder,  
                remove_existing_folder=True,
                **sorter_params  
            )
            
            found_units = sorting_result.get_unit_ids()
            print(f"     Found {len(found_units)} units in channel {chan_id}")
            
            for unit_idx, unit_id in enumerate(found_units):
                global_unit_id = f"{chan_id}_U{unit_idx}"
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
            
        finally:
            # IMMEDIATE CLEANUP: Delete the sorter folder for this channel
            if os.path.exists(chan_output_folder):
                shutil.rmtree(chan_output_folder, ignore_errors=True)

    # 5. EXPORT FINAL CONSOLIDATED DATA
    if len(all_spikes_data) > 0:
        print("\n=== All channels processed. Saving consolidated data ===")
        df_spikes = pd.DataFrame(all_spikes_data)
        
        df_spikes = df_spikes.sort_values(by='Spike_Time_Seconds').reset_index(drop=True)
        
        csv_path = os.path.join(output_folder, f"all_spikes_consolidated_{custom_name}.csv")
        df_spikes.to_csv(csv_path, index=False)
        
        print(f"Success! Total spikes found across all channels: {len(df_spikes)}")
        print(f"Data saved to: {csv_path}")
    else:
        print("\nNo spikes were found in any channel.")

    # 6. FINAL CACHE CLEANUP (OPTIONAL BASED ON USER DECISION)
    print("\n[+] Managing heavy temporary binary files...")
    # Unbind the variable and force Python to release the files
    del recording_saved   
    gc.collect()          
    
    if not keep_cached_binary:
        # User chose NOT to keep the folder (save space)
        if os.path.exists(cached_folder):
            try:
                shutil.rmtree(cached_folder, ignore_errors=True)
                print("    -> Cached binary deleted successfully. Storage space recovered!")
            except Exception as e:
                print(f"    -> Could not completely delete cached binary. Please remove it manually if needed. Error: {e}")
    else:
        # User chose YES to keep it
        print(f"    -> Cached binary KEPT at: {cached_folder}")
        print("    -> (Remember to delete it manually if you do not need it later)")