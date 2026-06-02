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
    
    # Assign the map depending on the file (assuming mcd uses the same as h5)
    map_key = "channel_mapping_rhs" if file_type == 'rhs' else "channel_mapping_h5"
    channel_map_list = mea_mapping[map_key]

    print(f"Assigning MEA spatial map for {num_channels} channels...")
    probe_mea = pi.Probe(ndim=2, si_units='um')
    positions, valid_channel_indices = [], []
    
    for i, num in enumerate(channel_map_list):
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
    
    # 1. SELECTION INTERFACE
    root = tk.Tk()
    root.withdraw()

    selected_file_paths = filedialog.askopenfilenames(
        title="Select recording files to PREPROCESS and EXPORT",
        filetypes=[
            ("All supported files", "*.h5 *.rhs *.mcd"), 
            ("H5 files", "*.h5"), 
            ("RHS files", "*.rhs"),
            ("MCD files", "*.mcd")
        ]
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
    
    # 2. LOADING AND CONCATENATION
    recording_list = []
    
    # Handling .h5 files
    if selected_file_paths[0].endswith('.h5'):
        print(f"Loading {len(selected_file_paths)} H5 file(s)...")
        for full_file_path in selected_file_paths:
            try:
                rec = se.read_mcsh5(full_file_path, stream_id='0')
                recording_list.append(rec)
            except Exception as e:
                print(f"  -> WARNING: Skipping '{os.path.basename(full_file_path)}'. Error: {e}")
                
    # Handling .rhs files
    elif selected_file_paths[0].endswith('.rhs'):
        print(f"Loading {len(selected_file_paths)} RHS file(s)...")
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            rec = spre.unsigned_to_signed(rec)
            recording_list.append(rec)
            
    # Handling .mcd files
    elif selected_file_paths[0].endswith('.mcd'):
        print(f"Loading {len(selected_file_paths)} MCD file(s)...")
        for full_file_path in selected_file_paths:
            try:
                # SpikeInterface uses neo.rawio.MCDRawIO under the hood
                rec = se.read_mcd(full_file_path)
                recording_list.append(rec)
            except Exception as e:
                print(f"  -> WARNING: Skipping '{os.path.basename(full_file_path)}'. Error: {e}")
    
    if not recording_list:
        print("\nError: No valid recordings were loaded. Operation canceled.")
        exit()

    recording = sc.concatenate_recordings(recording_list) if len(recording_list) > 1 else recording_list[0]
    num_channels = recording.get_num_channels()

    # Determine the file type to pass to the mapping
    if selected_file_paths[0].endswith('.h5'):
        file_type = 'h5'
    elif selected_file_paths[0].endswith('.rhs'):
        file_type = 'rhs'
    else:
        file_type = 'mcd'

    probe = create_probe(is_mea=True, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    # 3. PREPROCESSING (Noise cleaning)
    print("\nApplying bandpass filter (300-6000 Hz)...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    # 4. EXPORT FOR NATIVE TRIDESCLOUS
    fs = recording.get_sampling_frequency()
    export_folder = os.path.join(input_folder, f"tdc_ready_{custom_name}")
    print(f"\nExporting clean data to: {export_folder}")

    # Save recording in binary format (.raw)
    recording_saved = recording.save(folder=export_folder, format='binary', n_jobs=-1, overwrite=True)
    
    # Save electrode map in .prb format (native for Tridesclous)
    probegroup = pi.ProbeGroup()

    # Reset the indices from 0 to 58 to match the newly saved binary file
    probe.set_device_channel_indices(np.arange(recording_saved.get_num_channels()))
    probegroup.add_probe(probe)
    
    prb_path = os.path.join(export_folder, "mea_probe.prb")
    pi.write_prb(prb_path, probegroup)

    print("\n" + "="*50)
    print("EXPORT SUCCESSFULLY COMPLETED!")
    print("="*50)
    print("Your cleaned and mapped data is ready for the Tridesclous GUI.")
    print("\nWHEN YOU OPEN 'tdc' AND CREATE A NEW DATASET (Initialize Dataset), USE THESE PARAMETERS:")
    print(f"Format: Raw data")
    print(f"Data file (Filenames): {export_folder}/traces_cached_seg0.raw")
    print(f"Sample Rate: {fs}")
    print(f"Num Channels: {recording.get_num_channels()}")
    print(f"Data dtype: float32")
    print(f"In the 'Geometry' tab, load the file: {prb_path}")
    print("="*50)