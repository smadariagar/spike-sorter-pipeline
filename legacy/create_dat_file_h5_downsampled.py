import os
import gc
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog
import h5py

def get_channel_id_map(info_dset, max_channels=64):
    reorder_map = [-1] * max_channels
    try:
        id_col_name = None
        for col in ['ChannelID', 'UnitID', 'Label']: 
            if col in info_dset.dtype.names:
                id_col_name = col
                break
        
        if id_col_name is None:
            raise ValueError("Could not find ChannelID/UnitID/Label column")

        raw_ids = info_dset[id_col_name]
        print(f"     [INFO] Using column '{id_col_name}' for sorting.")

        for row_index, raw_id in enumerate(raw_ids):
            try:
                if isinstance(raw_id, bytes):
                    ch_id_int = int(raw_id.decode('utf-8'))
                else:
                    ch_id_int = int(raw_id)
                
                if 0 <= ch_id_int < max_channels:
                    reorder_map[ch_id_int] = row_index
            except ValueError:
                print(f"     [WARNING] Could not convert ID '{raw_id}' to integer. Ignoring.")
    except Exception as e:
        print(f"     [ERROR] Failed to create ID map: {e}")
        return None
    return reorder_map

def process_and_save_reordered(h5_dataset, reorder_map, output_path, chunk_size=200000, step=2):
    """
    Reads from H5 with a 'step' for downsampling, reorders rows, and saves to binary.
    step=2 effectively converts 40kHz to 20kHz by taking every other sample.
    """
    total_samples_raw = h5_dataset.shape[1]
    num_output_channels = len(reorder_map)
    CONVERSION_FACTOR = 10
    
    print(f"  -> Saving sorted binary (Downsampled 2x): {os.path.basename(output_path)}")
    
    with open(output_path, 'wb') as f_out:

        for start in range(0, total_samples_raw, chunk_size):
            end = min(start + chunk_size, total_samples_raw)
            
            raw_chunk = h5_dataset[:, start:end:step]
            
            current_chunk_width = raw_chunk.shape[1]
            
            # Empty matrix (Output Channels x Samples)
            ordered_chunk = np.zeros((num_output_channels, current_chunk_width), dtype=raw_chunk.dtype)
            
            for target_ch_idx, source_row_idx in enumerate(reorder_map):
                if source_row_idx != -1:
                    if source_row_idx < raw_chunk.shape[0]:
                        ordered_chunk[target_ch_idx, :] = raw_chunk[source_row_idx, :]
            
            ordered_chunk = ordered_chunk * CONVERSION_FACTOR
            np.clip(ordered_chunk, -32767, 32767, out=ordered_chunk)
            data_int16 = ordered_chunk.astype(np.int16)
            
            # Transpose to (Samples x Channels) for binary writing
            data_to_save = np.ascontiguousarray(data_int16.T)
            f_out.write(data_to_save.tobytes())
            
            if start % (chunk_size * 5) == 0:
                progress = (end / total_samples_raw) * 100
                print(f"     Progress (Input Read): {progress:.1f}%", end='\r')
            
            del raw_chunk, ordered_chunk, data_int16, data_to_save
            
    print(f"     Progress: 100.0% - Completed.")
    gc.collect()

##########################
# MAIN SCRIPT
##########################

root = tk.Tk()
root.withdraw()

selected_file_path = filedialog.askopenfilename(
    title="Select a .h5 file",
    filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
)

if not selected_file_path:
    print("Operation canceled.")
    exit() 

input_folder = os.path.dirname(selected_file_path)
output_folder = os.path.join(input_folder, "outputs_binary_20kHz") 
os.makedirs(output_folder, exist_ok=True)

search_pattern = os.path.join(input_folder, "*.h5")
h5_file_list = glob.glob(search_pattern)

print(f"Processing {len(h5_file_list)} files for 20kHz output...")

for idx, full_file_path in enumerate(h5_file_list):
    base_filename = os.path.basename(full_file_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_filename}")
    
    with h5py.File(full_file_path, 'r') as f:
        try:
            stream_group = f['Data']['Recording_0']['AnalogStream']['Stream_0']
            dset = stream_group['ChannelData']
            print(f"  H5 Original Shape: {dset.shape}")

            if 'InfoChannel' in stream_group:
                info_dset = stream_group['InfoChannel']
                full_map = get_channel_id_map(info_dset, max_channels=64)
                
                if full_map is None: continue

                # FILE 1: 0 to 31
                map_0_31 = full_map[0:32]
                path_out_1 = os.path.join(output_folder, f"{filename_no_ext}_chan_00_31_20kHz.000")
                process_and_save_reordered(dset, reorder_map=map_0_31, output_path=path_out_1, step=2)

                # FILE 2: 32 to 63
                map_32_63 = full_map[32:64]
                path_out_2 = os.path.join(output_folder, f"{filename_no_ext}_chan_32_63_20kHz.000")
                process_and_save_reordered(dset, reorder_map=map_32_63, output_path=path_out_2, step=2)
                
            else:
                print("  [CRITICAL ERROR] 'InfoChannel' not found.")

        except KeyError as e:
            print(f"  [ERROR] Incorrect H5 structure: {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")

print(f"\n{'='*60}\nDone.")