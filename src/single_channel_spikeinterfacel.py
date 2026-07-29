import pandas as pd
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import filedialog

import spikeinterface.core as sc
import spikeinterface_gui as sig

# =========================================================
# 1. SELECCIÓN DE CARPETA
# =========================================================
root = tk.Tk()
root.withdraw()

selected_folder = filedialog.askdirectory(
    title="Select the 'single_channel_sorting_...' folder",
    mustexist=True
)

if not selected_folder:
    print("Operation canceled.")
    exit()

# Buscar los archivos generados en el paso anterior
cached_folder = os.path.join(selected_folder, "cached_binary_full")
csv_files = [f for f in os.listdir(selected_folder) if f.startswith("all_spikes") and f.endswith(".csv")]

if not os.path.exists(cached_folder) or not csv_files:
    print("\n[ERROR] Missing files.")
    print(f"Could not find 'cached_binary_full' or the CSV file in: {selected_folder}")
    exit()

csv_path = os.path.join(selected_folder, csv_files[0])

# =========================================================
# 2. RECONSTRUCCIÓN DE DATOS
# =========================================================
print(f"Loading full recording from: {cached_folder} ...")
recording = sc.load(cached_folder)
fs = recording.get_sampling_frequency()

print(f"Loading spike times from: {csv_path} ...")
df = pd.read_csv(csv_path)

# Convertir el DataFrame (CSV) a un diccionario que SpikeInterface entienda
# Formato requerido: { 'ID_Neurona': array_de_frames, ... }
unit_dict = {}
for unit_id, group in df.groupby("Neuron_ID"):
    # Convertimos la columna de frames a un array de numpy (enteros)
    unit_dict[unit_id] = group["Spike_Frame"].to_numpy(dtype=int)

print(f"Total isolated units found across all channels: {len(unit_dict)}")

# Crear el objeto Sorting unificado
unified_sorting = sc.NumpySorting.from_unit_dict([unit_dict], sampling_frequency=fs)

# =========================================================
# 3. CREACIÓN DEL ANALIZADOR GLOBAL (SortingAnalyzer)
# =========================================================
analyzer_folder = os.path.join(selected_folder, "unified_analyzer")

# Si ya existía un análisis previo, lo borramos para evitar conflictos
if os.path.exists(analyzer_folder):
    print("Removing previous analyzer folder...")
    shutil.rmtree(analyzer_folder)

print("\nCreating Unified SortingAnalyzer...")
print("(This links the individual spikes back to the whole 60-channel geometry)")
analyzer = sc.create_sorting_analyzer(
    sorting=unified_sorting,
    recording=recording,
    format="binary_folder",
    folder=analyzer_folder
)

# =========================================================
# 4. EXTRACCIÓN DE FORMAS DE ONDA (COMPUTING EXTENSIONS)
# =========================================================
print("\nComputing metrics and waveforms (This might take a couple of minutes)...")
job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

# Para que la GUI funcione sin crashear, necesita calcular estas métricas mínimas:
analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
analyzer.compute("templates")
analyzer.compute("noise_levels")

# (Opcional) Si quieres ver el mapa 2D del PCA en la interfaz, descomenta esta línea:
# analyzer.compute("principal_components", n_components=3, mode='by_channel_local', **job_kwargs)

print("\nAnalyzer successfully created!")

# =========================================================
# 5. VISUAL INTERFACE
# =========================================================
print("\nOpening SpikeInterface GUI...")
app = sig.run_mainwindow(analyzer)