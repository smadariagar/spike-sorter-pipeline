import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def read_and_plot_binary(file_path, channel_idx, fs=20000, num_channels=32, duration_sec=1.0):
    """
    Reads a segment of a binary file and plots a specific channel.
    Assumes data is saved as (Samples x Channels) in int16.
    """
    
    # Calculate how many samples to read
    samples_to_read = int(fs * duration_sec)
    
    # Total integers to read = samples * channels
    count_total = samples_to_read * num_channels
    
    # Bytes per sample (int16 = 2 bytes)
    bytes_per_sample = 2
    
    try:
        # Open file in read-binary mode
        with open(file_path, 'rb') as f:
            # We assume we start at second 0. 
            # If you wanted to start later: f.seek(start_second * fs * num_channels * bytes_per_sample)
            
            # Read data
            raw_data = np.fromfile(f, dtype=np.int16, count=count_total)
            
        # Check if we read enough data
        if raw_data.size == 0:
            print("[ERROR] File is empty or could not be read.")
            return

        # Reshape: The file was saved as (Samples, Channels) flattened.
        # So we reshape to (Samples, Channels)
        data_matrix = raw_data.reshape(-1, num_channels)
        
        # Check if the requested channel is valid
        if channel_idx < 0 or channel_idx >= num_channels:
            print(f"[ERROR] Channel {channel_idx} is out of bounds (0-{num_channels-1}).")
            return

        # Extract the specific channel column
        channel_data = data_matrix[:, channel_idx]
        
        # Time vector
        t = np.linspace(0, duration_sec, len(channel_data))
        
        # Plotting
        # plt.figure(figsize=(10, 4))
        plt.plot(t, channel_data) # Green color
        # plt.title(f"File: {os.path.basename(file_path)}\nChannel Index (in file): {channel_idx}")
        # plt.xlabel("Time (seconds)")
        # plt.ylabel("Amplitude (int16 units)")
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"[ERROR] An issue occurred: {e}")

# --- MAIN ---

root = tk.Tk()
root.withdraw()

print("Please select the BINARY file (.000, .bin, etc)...")

# Open file dialog
file_path = filedialog.askopenfilename(
    title="Select a binary file",
    filetypes=[("Binary files", "*.000 *.bin *.dat"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

print(f"\nSelected file: {os.path.basename(file_path)}")

# Interactive Loop
plt.figure(figsize=(10, 4))
while True:
    print("\n" + "-"*40)
    user_input = input("Enter Channel Index (0-31) to plot (or 'q' to quit): ")

    if user_input.lower() == 'q':
        break
    
    try:
        ch_idx = int(user_input)
        # Call the plotting function
        read_and_plot_binary(file_path, ch_idx, fs=20000, num_channels=32, duration_sec=60.0)

        plt.show()
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")


print("Done.")