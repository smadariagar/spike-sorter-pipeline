import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.core as sc
import probeinterface as pi
import shutil  
import gc      

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 

# =========================================================
# PROBE 
# =========================================================
def create_probe(is_mea, file_type, num_channels, pitch=200, radius=15):
    if not is_mea:
        linear_probe = pi.generate_linear_probe(num_elec=num_channels, ypitch=20) 
        linear_probe.set_device_channel_indices(np.arange(num_channels))
        return linear_probe

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mea_mapping.json')
    with open(json_path, 'r') as f:
        mea_mapping = json.load(f)
    
    map_key = "channel_mapping_rhs" if file_type == 'rhs' else "channel_mapping_h5"
    list_2_map = mea_mapping[map_key]

    probe_mea = pi.Probe(ndim=2, si_units='um')
    positions, valid_channel_indices = [], []
    
    for i, num in enumerate(list_2_map):
        num_str = str(num)
        if num_str == '0':
            continue
            
        x = (int(num_str[0]) - 1) * pitch
        y = (8 - int(num_str[1])) * pitch
        
        positions.append([x, y])
        valid_channel_indices.append(i)

    probe_mea.set_contacts(
        positions=np.array(positions), 
        shapes='circle', 
        shape_params={'radius': radius}
    )

    probe_mea.set_device_channel_indices(valid_channel_indices)
    return probe_mea

# =========================================================
# FUNCIONES MATEMÁTICAS
# =========================================================
def find_elbow(inertias):
    n_points = len(inertias)
    if n_points < 3:
        return 0 

    p1 = np.array([1, inertias[0]])
    p2 = np.array([n_points, inertias[-1]])
    
    distances = []
    for i in range(n_points):
        p0 = np.array([i + 1, inertias[i]])
        num = np.abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        den = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
        distances.append(num / den)
        
    return np.argmax(distances) 

# =========================================================
# PARÁMETROS DEL SORTER PERSONALIZADO (K-MEANS)
# =========================================================
MEA_probe = True

custom_sorter_params = {
    'detect_threshold': 4.0,        
    'peak_sign': 'neg',             
    'exclude_sweep_ms': 1.5,        
    'max_clusters_to_test': 10,     
    'min_silhouette_score': 0.15    # Umbral mínimo puro
}

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    
    root = tk.Tk()
    root.withdraw()

    selected_file_paths = filedialog.askopenfilenames(
        title="Select recording files",
        filetypes=[("H5/RHS files", "*.h5 *.rhs"), ("All files", "*.*")]
    )

    if not selected_file_paths:
        exit() 

    selected_file_paths = sorted(list(selected_file_paths))

    custom_name = simpledialog.askstring("Output Name", "Enter the name for this analysis session:")
    if not custom_name:
        exit()

    keep_cached_binary = messagebox.askyesno(
        "Conservar Caché Binario", 
        "¿Deseas conservar la carpeta 'cached_binary_full' para usar Phy luego?"
    )

    show_plots = messagebox.askyesno(
        "Modo Interactivo (Human-in-the-Loop)", 
        "¿Deseas ver las 4 mejores divisiones de cada canal y ELEGIR MANUALMENTE la cantidad de clústeres?\n\n"
        "SÍ: Se pausará en cada canal para que veas los gráficos y decidas.\n"
        "NO: El algoritmo decidirá y procesará todo en segundo plano."
    )

    input_folder = os.path.dirname(selected_file_paths[0])
    output_folder = os.path.join(input_folder, f'single_channel_sorting/{custom_name}/')
    os.makedirs(output_folder, exist_ok=True)

    summary_txt_path = os.path.join(output_folder, f"analysis_summary_{custom_name}.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Session Name: {custom_name}\n")
        f.write(f"Algorithm: Custom Peak Detection + Interactive K-Means\n")
        for key, value in custom_sorter_params.items():
            f.write(f" * {key}: {value}\n")
    
    recording_list = []
    if selected_file_paths[0].endswith('.h5'):
        for full_file_path in selected_file_paths:
            try:
                recording_list.append(se.read_mcsh5(full_file_path, stream_id='0'))
            except Exception:
                pass
    elif selected_file_paths[0].endswith('.rhs'):
        for full_file_path in selected_file_paths:
            rec = se.read_intan(full_file_path, stream_id='0')
            recording_list.append(spre.unsigned_to_signed(rec))
    
    recording = sc.concatenate_recordings(recording_list) if len(recording_list) > 1 else recording_list[0]
    num_channels = recording.get_num_channels()
    
    file_type = 'h5' if selected_file_paths[0].endswith('.h5') else 'rhs'
    probe = create_probe(is_mea=MEA_probe, file_type=file_type, num_channels=num_channels)
    recording = recording.set_probe(probe)

    print("Aplicando filtro bandpass...")
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

    cached_folder = os.path.join(output_folder, "cached_binary_full")
    print(f"Guardando datos cacheados en disco...")
    job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)
    recording_saved = recording.save(folder=cached_folder, format='binary', overwrite=True, **job_kwargs)
    
    fs = recording_saved.get_sampling_frequency()

    # =========================================================
    # 4. CUSTOM SINGLE CHANNEL PIPELINE
    # =========================================================
    all_spikes_data = []
    channel_ids = recording_saved.get_channel_ids()
    print(f"\n=== Iniciando Detección y Clustering ===")

    for chan_id in channel_ids:
        print(f"\n---> Evaluando Canal: {chan_id}")
        rec_single_chan = recording_saved.select_channels(channel_ids=[chan_id])
        
        peaks = detect_peaks(
            recording=rec_single_chan,
            method='by_channel',
            peak_sign=custom_sorter_params['peak_sign'],
            detect_threshold=custom_sorter_params['detect_threshold'],
            exclude_sweep_ms=custom_sorter_params['exclude_sweep_ms']
        )
        
        spike_frames = peaks['sample_index']
        if len(spike_frames) < 10:
            print(f"     -> Muy pocos picos ({len(spike_frames)}). Se descarta clustering.")
            continue

        print(f"     -> {len(spike_frames)} picos aislados. Calculando PCA y Waveforms...")

        dummy_dict = {'DummyUnit': spike_frames}
        dummy_sorting = sc.NumpySorting.from_unit_dict([dummy_dict], sampling_frequency=fs)
        analyzer = sc.create_sorting_analyzer(sorting=dummy_sorting, recording=rec_single_chan, format="memory")
        analyzer.compute('random_spikes', max_spikes_per_unit=500000) 
        analyzer.compute('waveforms', ms_before=1.0, ms_after=2.0)
        analyzer.compute('principal_components', n_components=3, mode='by_channel_local')
        
        pca_ext = analyzer.get_extension('principal_components')
        pcs = pca_ext.get_projections_one_unit('DummyUnit')
        X = pcs.reshape(pcs.shape[0], -1)

        wf_ext = analyzer.get_extension('waveforms')
        wfs = wf_ext.get_waveforms_one_unit('DummyUnit')
        best_chan_idx = np.argmin(np.min(np.mean(wfs, axis=0), axis=0))

        max_k_to_test = min(custom_sorter_params['max_clusters_to_test'], len(X) - 1)
        
        inertias = []
        silhouettes = []
        kmeans_models = []

        for k in range(1, max_k_to_test + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            kmeans_models.append(labels)
            
            if k == 1:
                silhouettes.append(0.0)
            else:
                silhouettes.append(silhouette_score(X, labels))
                
        # --- CÁLCULO DE MÉTRICAS PURAS ---
        elbow_idx = find_elbow(inertias)
        best_elbow_k = elbow_idx + 1
        
        max_sil_idx = np.argmax(silhouettes)
        best_sil_k = max_sil_idx + 1
        max_sil_score = silhouettes[max_sil_idx]

        print(f"     -> Codo K={best_elbow_k} | Sugerencia Silueta Pura K={best_sil_k} (Score: {max_sil_score:.2f})")

        final_k = 1
        
        # Pre-calculamos la decisión de la máquina (puramente basada en la silueta)
        if max_sil_score >= custom_sorter_params['min_silhouette_score']:
            final_k = best_sil_k

        # ---> BLOQUE DE VISUALIZACIÓN Y DECISIÓN HUMANA (2x3)
        if show_plots:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Gráfico 1: Inercia (Codo) - Fila 0, Columna 0
            axes[0, 0].plot(range(1, max_k_to_test + 1), inertias, marker='o', color='royalblue')
            axes[0, 0].axvline(x=best_elbow_k, color='red', linestyle='--', label=f'Codo (K={best_elbow_k})')
            axes[0, 0].set_title('Método del Codo (Inercia)')
            axes[0, 0].set_xlabel('Número de Clústeres (K)')
            axes[0, 0].set_ylabel('Inercia')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # Gráfico 2: Silueta Pura - Fila 0, Columna 1
            if max_k_to_test > 1:
                axes[0, 1].plot(range(2, max_k_to_test + 1), silhouettes[1:], marker='o', color='forestgreen', label='Silueta')
                axes[0, 1].axvline(x=best_sil_k, color='red', linestyle='-', label=f'Sugerencia (K={best_sil_k})')
                axes[0, 1].axhline(y=custom_sorter_params['min_silhouette_score'], color='black', linestyle=':', label='Umbral')
            axes[0, 1].set_title('Puntuación de Silueta')
            axes[0, 1].set_xlabel('Número de Clústeres (K)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            # Gráficos 3 a 6: Visualización de K=1, K=2, K=3 y K=4
            k_views = [1, 2, 3, 4]
            ax_views = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

            for k_val, ax in zip(k_views, ax_views):
                if k_val <= max_k_to_test:
                    lbls = kmeans_models[k_val - 1]
                    for cluster_label in range(k_val):
                        idx = (lbls == cluster_label)
                        if np.any(idx):
                            cluster_wfs = wfs[idx, :, best_chan_idx]
                            template = np.mean(cluster_wfs, axis=0)
                            ax.plot(template, label=f'C{cluster_label} (n={np.sum(idx)})', linewidth=2)
                    ax.set_title(f'¿Cortar en {k_val} neurona(s)?')
                    ax.set_xlabel('Muestras')
                    ax.set_ylabel('Amplitud (V)' if k_val in [1, 2] else '')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, loc='upper right')
                else:
                    ax.axis('off') 

            plt.suptitle(f'Decisión Interactiva - Canal: {chan_id} | Total Espigas: {len(X)}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show() 

            # ---> LA PREGUNTA AL USUARIO
            user_decision = simpledialog.askinteger(
                "Selección Manual de Clústeres", 
                f"Evaluación del Canal {chan_id}\n\n"
                f"Sugerencia matemática pura (Silueta): {final_k} clúster(es).\n\n"
                f"Mirando los gráficos, ¿En cuántas neuronas deseas dividir este canal realmente?",
                initialvalue=final_k, 
                minvalue=1,
                maxvalue=10
            )

            if user_decision is not None:
                final_k = user_decision
                print(f"     -> [HUMANO] Decidiste forzar la separación en {final_k} unidades.")
            else:
                print(f"     -> [MÁQUINA] Se usará la sugerencia automática: {final_k} unidades.")
        
        else:
            if final_k > 1:
                print(f"     -> [MÁQUINA] Separación aceptada en {final_k} unidades.")
            else:
                print(f"     -> [MÁQUINA] Clústeres superpuestos. 1 sola neurona.")

        final_labels = kmeans_models[final_k - 1]

        # GUARDAR RESULTADOS CLASIFICADOS EN LA LISTA
        for i, frame in enumerate(spike_frames):
            global_unit_id = f"Ch{chan_id}_U{final_labels[i]}"
            all_spikes_data.append({
                'Electrode_ID': chan_id,
                'Neuron_ID': global_unit_id,
                'Spike_Frame': frame,
                'Spike_Time_Seconds': frame / fs
            })

    # 5. EXPORT FINAL CONSOLIDATED DATA
    if len(all_spikes_data) > 0:
        print("\n=== All channels processed. Saving consolidated data ===")
        df_spikes = pd.DataFrame(all_spikes_data)
        df_spikes = df_spikes.sort_values(by='Spike_Time_Seconds').reset_index(drop=True)
        csv_path = os.path.join(output_folder, f"all_spikes_consolidated_{custom_name}.csv")
        df_spikes.to_csv(csv_path, index=False)
        print(f"Success! Total spikes isolated and clustered: {len(df_spikes)}")
    else:
        print("\nNo spikes were found in any channel.")

    # 6. LIMPIEZA
    del recording_saved   
    gc.collect()          
    
    if not keep_cached_binary and os.path.exists(cached_folder):
        shutil.rmtree(cached_folder, ignore_errors=True)
        print("\n[+] Cached binary deleted successfully. Storage space recovered!")
        