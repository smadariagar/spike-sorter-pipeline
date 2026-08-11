import os
import pandas as pd

def transform_boss_sorting(excel_path, output_csv_path, fs=40000):
    # Cargar archivo Excel sin encabezado
    df_raw = pd.read_excel(excel_path, header=None)
    
    # Extraer ID combinado y frames
    raw_ids = df_raw[0]
    frames = df_raw[1]
    
    # Separar canal y unidad (últimos 2 dígitos = unidad, dígitos previos = electrodo)
    electrodes = raw_ids // 100
    units = raw_ids % 100
    
    # Corrección de +64 a +32 (restar 32 a e > 32) Y restar 1 para indexación base 0 (0 a 58)
    corrected_electrodes = electrodes.apply(lambda e: (e - 32 if e > 32 else e) - 1)
    
    # Reconstruir el DataFrame
    df_transformed = pd.DataFrame({
        'Electrode_ID': ['Ch' + str(e) for e in corrected_electrodes],
        'Neuron_ID': ['Ch' + str(e) + '_U' + str(u) for e, u in zip(corrected_electrodes, units)],
        'Spike_Frame': frames,
        'Spike_Time_Seconds': frames / fs
    })
    
    # Ordenar cronológicamente por Spike_Frame
    df_transformed = df_transformed.sort_values(by='Spike_Frame').reset_index(drop=True)
    
    # Guardar CSV
    df_transformed.to_csv(output_csv_path, index=False)
    print(f"Archivo guardado exitosamente en: {output_csv_path}")

# Ejemplo de uso:

path_xlsx = '/home/samuel/Documentos/Explora/spike_sorter'
transform_boss_sorting(os.path.join(path_xlsx,'MEA36 A.xlsx'), os.path.join(path_xlsx,'all_spikes_consolidated_MEA36_A.csv'))