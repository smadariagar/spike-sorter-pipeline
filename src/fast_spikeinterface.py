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
# HELPER FUNCTIONS
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

    # Safety check
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
# GENERAL AND SORTER PARAMETERS (EDIT ZONE)
# =========================================================

# General settings
MEA_probe = True
sorter_name = 'mountainsort5' 

# Sorter parameters dictionary
# You can modify any parameter here without touching the program's logic
sorter_params = {
    'detect_threshold': 3.0,                            # Sensitivity for detecting spikes (previously 3.0)
    'detect_sign': -1,                                  # -1 looks for negative peaks (standard extracellular)
    'n_jobs': -1,                                       # -1 uses all CPU cores (num_workers for MS4 and n_jobs for MS5)
    'filter': False,                                    # IMPORTANT: False because we already filter in Phase 2
    'whiten': True,                                     # Spatial noise whitening
    'scheme2_training_duration_sec': 600,               # 
    'scheme2_max_num_snippets_per_training_batch': 500,
    'npca_per_channel': 5,                              # FORCE AMPLITUDE SEPARATION
    'snippet_T1': 30,                                   # Extend the analysis window before the peak
    'snippet_T2': 30
}

# ==============================================================================
# MOUNTAINSORT 5: COMPLETE PARAMETER REFERENCE
# ==============================================================================

# 1. Detection Parameters (Signal Search)
# ------------------------------------------------------------------------------
# 'detect_threshold' (5.5): Statistical threshold. The spike must be 5.5 times 
#                           larger than the Median Absolute Deviation (background 
#                           noise) to be detected.
# 'detect_sign' (-1):       Spike direction. -1 looks for valleys (negative peaks, 
#                           standard for extracellular recordings), 1 looks for 
#                           positive peaks, 0 looks for both.
# 'detect_time_radius_msec' (0.5): Blind refractory period (in milliseconds). 
#                                  After detecting a peak, it ignores extra 
#                                  fluctuations for half a millisecond to avoid 
#                                  counting the same spike twice.

# 2. Waveform Parameters
# ------------------------------------------------------------------------------
# 'snippet_T1' (20) & 'snippet_T2' (20): Defines the time window the algorithm 
#                                        "snips" before (T1) and after (T2) the 
#                                        negative peak to analyze the spike shape. 
#                                        20 samples depend on your sampling frequency.
# 'snippet_mask_radius' (250): Radius (in micrometers). Defines how far it will 
#                              look in neighboring electrodes to reconstruct the 
#                              full signal of the same neuron across space.

# 3. Mathematical and Clustering Parameters
# ------------------------------------------------------------------------------
# 'scheme' ('2'): MS5 has different training phases. Scheme 2 is the recommended 
#                 one because it divides the process into detection, feature 
#                 training, and clustering.
# 'npca_per_channel' (3): Principal Components per channel. The algorithm 
#                         summarizes the waveform into 3 key mathematical values. 
#                         (Increase to 4 or 5 to separate spikes of different amplitudes).
# 'scheme1_...', 'scheme2_...': Detection Radii. Defines how large the electrode 
#                               "neighborhoods" are that are analyzed together in 
#                               each training phase.
# 'scheme2_training_duration_sec' (300): Takes 5 minutes of your data to learn 
#                                        what the neurons look like and build 
#                                        the initial clusters.
# 'scheme2_training_recording_sampling_mode' ('uniform'): Extracts those 5 minutes 
#                                                         by taking small snippets 
#                                                         from the entire file 
#                                                         uniformly, not just from 
#                                                         the first 5 minutes.

# 4. Built-in Filters (Preprocessing)
# ------------------------------------------------------------------------------
# 'filter' (True): Applies an internal bandpass filter. (Remember to set it to 
#                  False in your code if you already do this in Phase 2).
# 'freq_min' (300) & 'freq_max' (6000): The limits of that filter in Hz.
# 'whiten' (True): Spatial whitening. Reduces the background noise shared between 
#                  neighboring electrodes.

# 5. Performance and Computing
# ------------------------------------------------------------------------------
# 'n_jobs' (1): BE CAREFUL HERE! MS5 went back to calling this parameter 'n_jobs' 
#               (unlike MS4, which used 'num_workers'). Set it to -1 in your 
#               dictionary to use all your CPU cores.
# 'chunk_duration' ('1s'): Loads the recording into RAM in 1-second chunks to 
#                          prevent your PC from crashing.
# 'delete_temporary_recording' (True): Deletes large temporary files after finishing.

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
        print(f"Loading {len(selected_file_paths)} H5 file(s) (Intan/MCS)...")
        for full_file_path in selected_file_paths:
            try:
                rec = se.read_mcsh5(full_file_path, stream_id='0')
                recording_list.append(rec)
            except IndexError:
                print(f"  -> WARNING: Skipping '{os.path.basename(full_file_path)}' (Likely empty or corrupted file)")
            except Exception as e:
                print(f"  -> WARNING: Could not load '{os.path.basename(full_file_path)}'. Error: {e}")
     
    elif selected_file_paths[0].endswith('.rhs'):
        print(f"Loading {len(selected_file_paths)} RHS file(s) (Intan)...")
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            rec = spre.unsigned_to_signed(rec) # true zero
            recording_list.append(rec)
    
    if not recording_list:
        print("\nError: No valid recordings were loaded. Operation canceled.")
        exit()

    # One o more files
    if len(recording_list) > 1:
        recording = sc.concatenate_recordings(recording_list)
    else:
        recording = recording_list[0]

    num_channels = recording.get_num_channels()
    print(f"Loaded channels: {num_channels}")

    # Use the helper function to assign the spatial map
    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'

    probe = create_probe(is_mea=MEA_probe, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    print(f"Effective channels for sorting (Grounds removed): {recording.get_num_channels()}")

    # =========================================================
    # PHASE 2: SPIKE SORTING
    # =========================================================
    print("Applying chained preprocessing...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    print(f"\nStarting {sorter_name}...")
    
    # Run the sorter injecting the dictionary (**sorter_params)
    sorting_result = ss.run_sorter(
        sorter_name=sorter_name,
        recording=recording,
        folder=sorting_output_folder,  
        remove_existing_folder=True,
        **sorter_params  # sorter parameters here
    )

    found_units = sorting_result.get_unit_ids()
    print("\n=== Sorting Finished ===")
    print(f"Potential neurons (clusters) found: {len(found_units)}")
    print(f"Unit IDs: {found_units}")

    # =========================================================
    # PHASE 3: WAVEFORM EXTRACTION
    # =========================================================
    print("\nCreating the waveform analyzer (SortingAnalyzer)...")
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