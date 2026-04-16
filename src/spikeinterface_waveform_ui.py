import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

import spikeinterface.core as sc
import spikeinterface_gui as sig

print("Loading data analyzer...")

root = tk.Tk()
root.withdraw()

# FOLDER SELECTION
# Added 'mustexist=True' and a much clearer title
selected_folder_path = filedialog.askdirectory(
    title="Select the specific 'waveforms_...' folder (Click once and press OK)",
    mustexist=True
)

if not selected_folder_path:
    print("Operation canceled.")
    exit() 

print(f"Attempting to load analyzer from: {selected_folder_path}")

# VALIDATION AND ERROR HANDLING
try:
    analyzer = sc.load_sorting_analyzer(selected_folder_path)
except Exception as e:
    print("\nERROR: Could not load the analyzer.")
    print("Make sure you selected the EXACT folder containing the waveforms data")
    print("(e.g., 'waveforms_my_recording'), and not the parent 'sorter_results' folder.")
    print(f"Technical details: {e}")
    exit()

# =========================================================
# CSV EXPORT
# =========================================================
print("\nExtracting list of spikes and channels for CSV...")

# 'sorting' object and sampling frequency from the analyzer
sorting = analyzer.sorting
fs = sorting.get_sampling_frequency()
found_units = sorting.get_unit_ids()

# Calculate the main channel for each unit
main_channels = sc.get_template_extremum_channel(analyzer)

# Collect data
spike_data = []
for unit_id in found_units:
    channel = main_channels[unit_id]
    
    spike_train_samples = sorting.get_unit_spike_train(unit_id)
    spike_train_seconds = spike_train_samples / fs

    for time_s in spike_train_seconds:
        spike_data.append({
            'Unit_ID': unit_id,
            'Channel': channel,
            'Time_s': time_s
        })

# DataFrame
df_spikes = pd.DataFrame(spike_data)

if not df_spikes.empty:
    df_spikes = df_spikes.sort_values(by='Time_s').reset_index(drop=True)
    
    csv_file = os.path.join(selected_folder_path, "spike_list.csv")
    df_spikes.to_csv(csv_file, index=False)
    
    print(f"Success! Saved {len(df_spikes)} spikes.")
    print(f"CSV file saved at: {csv_file}")
else:
    print("No spikes found to export.")

# =========================================================
# VISUAL INTERFACE
# =========================================================
print("\nOpening the visual interface...")

# Launch the application
app = sig.run_mainwindow(analyzer)