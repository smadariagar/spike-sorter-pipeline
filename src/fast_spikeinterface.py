import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.widgets as sw
import probeinterface as pi

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def create_probe(is_mea, num_channels, pitch=100, radius=15):
    """
    Generates the electrode map depending on the geometry and number of channels.
    """
    if not is_mea:
        print("Assigning linear spatial electrode map...")
        linear_probe = pi.generate_linear_probe(num_elec=num_channels, ypitch=20) 
        linear_probe.set_device_channel_indices(np.arange(num_channels))
        return linear_probe

    print(f"Assigning MEA spatial map for {num_channels} channels...")
    probe_mea = pi.Probe(ndim=2, si_units='um')
    positions = []
    
    for y in range(8):
        for x in range(8):
            # If the device has 60 channels, omit the 4 corners of the 8x8 grid
            if num_channels == 60:
                if (x == 0 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 0) or (x == 7 and y == 7):
                    continue
            # If it has 64 channels, this condition is not met and saves all 64 points
            
            positions.append([x * pitch, y * pitch])

    # Safety check
    if len(positions) != num_channels:
        raise ValueError(f"Hardware mismatch: Tried to map {len(positions)} positions to {num_channels} channels.")

    probe_mea.set_contacts(
        positions=np.array(positions), 
        shapes='circle', 
        shape_params={'radius': radius}
    )
    probe_mea.set_device_channel_indices(np.arange(num_channels))
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
    'detect_threshold': 5.5,   # Sensitivity for detecting spikes (previously 3.0)
    'detect_sign': -1,         # -1 looks for negative peaks (standard extracellular)
    'n_jobs': -1,              # -1 uses all CPU cores (Changed from num_workers for MS5)
    'filter': False,           # IMPORTANT: False because we already filter in Phase 2
    'whiten': True             # Spatial noise whitening
}

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    # 1. FILE SELECTION UI
    root = tk.Tk()
    root.withdraw()
    selected_file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("H5 files", "*.h5"), ("RHS files", "*.rhs"), ("All files", "*.*")]
    )

    if not selected_file_path:
        print("Operation canceled.")
        exit() 

    input_folder = os.path.dirname(selected_file_path)
    file_name = os.path.basename(selected_file_path)
    output_folder = os.path.join(input_folder, 'sorter_results/')
    os.makedirs(output_folder, exist_ok=True)

    # =========================================================
    # PHASE 1: DATA LOADING AND GEOMETRY
    # =========================================================
    if file_name[-2:] == 'h5':
        print("Loading data from H5 file (Intan)...")
        search_pattern = os.path.join(input_folder, "*.h5")
        file_list = sorted(glob.glob(search_pattern))        
        recording_list = []
        for full_file_path in file_list:
            rec = se.read_mcsh5(full_file_path, stream_id='0')
            recording_list.append(rec)

        sorting_output_folder = os.path.join(output_folder, "sorting_" + file_name[:-3]) 
        waveforms_folder = os.path.join(output_folder, "waveforms_" + file_name[:-3]) 
        recording = sc.concatenate_recordings(recording_list)

    elif file_name[-3:] == 'rhs':
        print("Loading data from RHS file (Intan)...")
        recording = se.read_intan(selected_file_path, stream_id='0')
        recording = spre.unsigned_to_signed(recording) # true zero
        sorting_output_folder = os.path.join(output_folder, "sorting_" + file_name[:-4]) 
        waveforms_folder = os.path.join(output_folder, "waveforms_" + file_name[:-4]) 

    num_channels = recording.get_num_channels()
    print(f"Loaded channels: {num_channels}")

    # Use the helper function to assign the spatial map
    probe = create_probe(is_mea=MEA_probe, num_channels=num_channels)
    recording = recording.set_probe(probe)

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
        **sorter_params  # <--- Unpacks all your parameters here
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