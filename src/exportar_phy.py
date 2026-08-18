import pandas as pd
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import filedialog
import spikeinterface.core as sc
import spikeinterface.exporters as sexp

# =========================================================
# 1. SELECCIÓN DE ARCHIVOS (UI)
# =========================================================
print("Abriendo selector de archivos...")
root = tk.Tk()
root.withdraw()

# El usuario solo necesita seleccionar el archivo de texto de resumen
summary_file_path = filedialog.askopenfilename(
    title="Selecciona el archivo 'analysis_summary_...txt'",
    filetypes=[("Text files", "*.txt")]
)

if not summary_file_path:
    print("Operación cancelada por el usuario.")
    exit()

selected_folder = os.path.dirname(summary_file_path)

# Buscar los archivos generados por tu Script 1
cached_folder = os.path.join(selected_folder, "cached_binary_full")
csv_files = [f for f in os.listdir(selected_folder) if f.startswith("all_spikes") and f.endswith(".csv")]

if not os.path.exists(cached_folder) or not csv_files:
    print(f"\n[ERROR] Faltan archivos críticos en: {selected_folder}")
    print("Asegúrate de que la carpeta 'cached_binary_full' y el archivo .csv existan.")
    exit()

csv_path = os.path.join(selected_folder, csv_files[0])

# =========================================================
# 2. RECONSTRUCCIÓN DE DATOS
# =========================================================
print(f"\n[1/4] Cargando el registro completo desde la caché binaria...")
try:
    recording = sc.load(cached_folder)
except Exception as e:
    print(f"\n[ERROR] No se pudo cargar el registro. Detalles: {e}")
    exit()
    
fs = recording.get_sampling_frequency()

print(f"[2/4] Cargando los tiempos de espigas desde el CSV...")
df = pd.read_csv(csv_path)

# Convertir el DataFrame de vuelta a un diccionario que SpikeInterface entienda
unit_dict = {}
for unit_id, group in df.groupby("Neuron_ID"):
    unit_dict[unit_id] = group["Spike_Frame"].to_numpy(dtype=int)

print(f"      -> Unidades totales aisladas: {len(unit_dict)}")

# Crear el objeto Sorting unificado
unified_sorting = sc.NumpySorting.from_unit_dict([unit_dict], sampling_frequency=fs)

# =========================================================
# 3. CREACIÓN DEL ANALIZADOR GLOBAL (SortingAnalyzer)
# =========================================================
analyzer_folder = os.path.join(selected_folder, "unified_analyzer")

if os.path.exists(analyzer_folder):
    print("      -> Borrando carpeta de analizador anterior para evitar conflictos...")
    shutil.rmtree(analyzer_folder)

print("\n[3/4] Construyendo el SortingAnalyzer...")
print("      (Este paso cruza los tiempos de las espigas con la geometría de los 60 canales)")
analyzer = sc.create_sorting_analyzer(
    sorting=unified_sorting,
    recording=recording,
    format="binary_folder",
    folder=analyzer_folder
)

# Phy requiere ciertas métricas para funcionar correctamente
print("      -> Calculando métricas y formas de onda (Waveforms)...")
analyzer.compute('random_spikes', method='uniform', max_spikes_per_unit=500, save=True)
analyzer.compute('waveforms', save=True)
analyzer.compute('templates', save=True)
analyzer.compute('noise_levels', save=True)

# Este es el paso crítico para ver las nubes de puntos y separar clústeres a mano
print("      -> Calculando Componentes Principales (PCA)...")
analyzer.compute('principal_components', n_components=6, mode='by_channel_local', save=True)

# =========================================================
# 4. EXPORTACIÓN A PHY
# =========================================================
phy_output_path = os.path.join(selected_folder, "phy_export")

print(f"\n[4/4] Exportando datos a formato Phy en: {phy_output_path}")
print("      Esto tomará un par de minutos, por favor espera...")

try:
    # copy_binary=True asegura que Phy cree su propia copia independiente de los datos
    sexp.export_to_phy(analyzer, output_folder=phy_output_path, remove_if_exists=True, copy_binary=True)
    
    print("\n=========================================================")
    print("                  ¡EXPORTACIÓN EXITOSA!                  ")
    print("=========================================================\n")
    print("Para comenzar la curación manual, abre tu terminal y ejecuta:")
    print(f'\nphy template-gui "{os.path.join(phy_output_path, "params.py")}"\n')
    
    print("NOTA: Ahora que Phy tiene su propia copia de los datos,")
    print("puedes borrar manualmente la carpeta pesada 'cached_binary_full' si necesitas liberar espacio.")

except Exception as e:
    print(f"\n[ERROR] Ocurrió un fallo durante la exportación a Phy: {e}")


# phy template-gui "/home/samuel/Documentos/Explora/spike_sorter/data/MEA36/single_channel_sorting/A_params_mnt5/phy_export/params.py"