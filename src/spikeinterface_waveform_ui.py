import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

import spikeinterface.core as sc
import spikeinterface_gui as sig

print("Loading data analyzer...")

root = tk.Tk()
root.withdraw()

selected_folder_path = filedialog.askdirectory(
    title="Select a folder",
)

if not selected_folder_path:
    print("Operation canceled.")
    exit() 

analyzer = sc.load_sorting_analyzer(selected_folder_path)

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

# VISUAL INTERFACE
# =========================================================
print("\nOpening the visual interface...")
# Launch the application
app = sig.run_mainwindow(analyzer)