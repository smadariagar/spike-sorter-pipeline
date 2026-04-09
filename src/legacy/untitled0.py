#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:38:45 2026

@author: pedro
"""

import tkinter as tk
from tkinter import filedialog
import struct

def select_and_read_file():
    # 1. Create a hidden root window for tkinter
    root = tk.Tk()
    root.withdraw()

    # 2. Open the Windows file picker dialog
    file_path = filedialog.askopenfilename(
        title="Select a binary file",
        filetypes=[("All files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        # 3. Open the file in binary read mode ('rb')
        with open(file_path, 'rb') as f:
            # A short int is 2 bytes. To read 10 of them, we need 20 bytes.
            num_elements = 100
            bytes_per_short = 2
            data = f.read(num_elements * bytes_per_short)

            if len(data) < num_elements * bytes_per_short:
                print(f"Warning: File is too small. Only read {len(data)} bytes.")
                # Adjust number of elements based on actual bytes read
                num_elements = len(data) // bytes_per_short

            if num_elements == 0:
                print("No data to read.")
                return

            # 4. Unpack the binary data
            # '<' indicates little-endian (standard for Windows/Intel)
            # 'h' indicates a signed short (2 bytes)
            # We use f'{num_elements}h' to read the specific count
            format_string = f'<{num_elements}h'
            short_ints = struct.unpack(format_string, data[:num_elements * bytes_per_short])

            print(f"First {num_elements} short integers:")
            for i, val in enumerate(short_ints):
                print(f"Index {i}: {val}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    select_and_read_file()