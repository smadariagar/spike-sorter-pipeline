import os
import gc
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog
import h5py
from scipy import signal

def get_channel_id_map(info_dset, max_channels=64):
    """
    Creates a map that relates: ChannelID (Target) -> RowIndex (Source in H5).
    Returns a list of length 'max_channels'.
    """
    reorder_map = [-1] * max_channels
    try:
        id_col_name = None
        # Check for common MCS column names
        for col in ['ChannelID', 'UnitID', 'Label']: 
            if col in info_dset.dtype.names:
                id_col_name = col
                break
        
        if id_col_name is None:
            raise ValueError("Could not find ID column (ChannelID/UnitID/Label)")

        raw_ids = info_dset[id_col_name]
        for row_index, raw_id in enumerate(raw_ids):
            try:
                # Handle both byte strings and integers
                ch_id_int = int(raw_id.decode('utf-8')) if isinstance(raw_id, bytes) else int(raw_id)
                if 0 <= ch_id_int < max_channels:
                    reorder_map[ch_id_int] = row_index
            except ValueError:
                continue
        
        return reorder_map 

    except Exception as e:
        print(f"     [ERROR] Failed to create ID map: {e}")
        return None

def process_and_save_reordered(h5_dataset, reorder_map, output_path, fs_in, fs_out, chunk_size=400000):
    """
    Reads from H5, reorders channels, applies Polyphase Resampling, 
    and saves to int16 binary.
    """
    total_samples = h5_dataset.shape[1]
    num_output_channels = len(reorder_map)
    CONVERSION_FACTOR = 10 
    
    # Resampling parameters
    up = int(fs_out)
    down = int(fs_in)
    
    print(f"  -> Saving sorted binary: {os.path.basename(output_path)}")
    if fs_in != fs_out:
        print(f"     [RESAMPLE] From {fs_in} Hz to {fs_out} Hz")

    with open(output_path, 'wb') as f_out:
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            current_chunk_width = end - start
            
            # 1. Read and Reorder
            raw_chunk = h5_dataset[:, start:end]
            ordered_chunk = np.zeros((num_output_channels, current_chunk_width), dtype=raw_chunk.dtype)
            
            for target_ch_idx, source_row_idx in enumerate(reorder_map):
                if source_row_idx != -1 and source_row_idx < raw_chunk.shape[0]:
                    ordered_chunk[target_ch_idx, :] = raw_chunk[source_row_idx, :]
            
            # 2. Scientific Resampling
            if fs_in != fs_out:
                # signal.resample_poly applies an automatic anti-aliasing filter
                ordered_chunk = signal.resample_poly(ordered_chunk, up, down, axis=1)
            
            # 3. Scaling and Clipping
            # Ensure values stay within the signed 16-bit integer range: [-32768, 32767]
            ordered_chunk = ordered_chunk * CONVERSION_FACTOR
            np.clip(ordered_chunk, -32767, 32767, out=ordered_chunk)
            
            # 4. Cast to int16 (2 bytes per sample for C++ compatibility)
            data_int16 = ordered_chunk.astype(np.int16)
            
            # 5. Transpose to (Samples x Channels) and ensure memory contiguity
            data_to_save = np.ascontiguousarray(data_int16.T)
            f_out.write(data_to_save.tobytes())
            
            if start % (chunk_size * 2) == 0:
                progress = (end / total_samples) * 100
                print(f"     Progress: {progress:.1f}%", end='\r')
            
            # Explicitly free memory
            del raw_chunk, ordered_chunk, data_int16, data_to_save
            
    print(f"     Progress: 100.0% - Completed.")
    gc.collect()

##########################
# MAIN EXECUTION
##########################

# Sampling Rates configuration
FS_ORIGINAL = 40000.0
FS_TARGET   = 20000.0  # Set equal to FS_ORIGINAL to skip resampling

root = tk.Tk()
root.withdraw()

selected_file_path = filedialog.askopenfilename(
    title="Select an .h5 file",
    filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
)

if not selected_file_path:
    print("Operation canceled.")
    exit() 

input_folder = os.path.dirname(selected_file_path)
output_folder = os.path.join(input_folder, f"outputs_bin_{FS_TARGET}Hz") 
os.makedirs(output_folder, exist_ok=True)

h5_file_list = glob.glob(os.path.join(input_folder, "*.h5"))
print(f"Found {len(h5_file_list)} files to process.")

for idx, full_file_path in enumerate(h5_file_list):
    base_filename = os.path.basename(full_file_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_filename}")
    
    with h5py.File(full_file_path, 'r') as f:
        try:
            # Navigate to the standard MCS H5 structure
            stream_group = f['Data']['Recording_0']['AnalogStream']['Stream_0']
            dset = stream_group['ChannelData']
            
            if 'InfoChannel' in stream_group:
                info_dset = stream_group['InfoChannel']
                full_map = get_channel_id_map(info_dset, max_channels=64)
                
                if full_map is None: continue

                # File 1: Channel IDs 0-31
                path_out_1 = os.path.join(output_folder, f"{filename_no_ext}_ch00_31.dat")
                process_and_save_reordered(
                    dset, 
                    reorder_map=full_map[0:32], 
                    output_path=path_out_1,
                    fs_in=FS_ORIGINAL,
                    fs_out=FS_TARGET
                )

                # File 2: Channel IDs 32-63
                path_out_2 = os.path.join(output_folder, f"{filename_no_ext}_ch32_63.dat")
                process_and_save_reordered(
                    dset, 
                    reorder_map=full_map[32:64], 
                    output_path=path_out_2,
                    fs_in=FS_ORIGINAL,
                    fs_out=FS_TARGET
                )
            else:
                print("  [ERROR] 'InfoChannel' not found in H5 structure.")

        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")

print(f"\n{'='*60}\nDone.")