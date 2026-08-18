import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import spikeinterface.core as sc
import tkinter as tk
from tkinter import filedialog

# =========================================================
# FUNCION SISTEMÁTICA DE K-MEANS (K=2 a 10) CON MÉTRICAS
# =========================================================
def explorar_splits_kmeans_sistematico(analyzer, unit_id, max_clusters=10):
    print(f"\n--- Evaluando particiones K-Means para la unidad: {unit_id} ---")
    
    # 1. Extraer extensiones
    try:
        wf_ext = analyzer.get_extension("waveforms")
        pca_ext = analyzer.get_extension("principal_components")
    except KeyError:
        print("\n[ERROR] El analizador debe tener computados 'waveforms' y 'principal_components'.")
        return

    if wf_ext is None or pca_ext is None:
        print("\n[ERROR] Faltan datos de PCA o Waveforms en este analizador.")
        return

    # Extraer los datos crudos
    try:
        wfs = wf_ext.get_waveforms_one_unit(unit_id)  
        pcs = pca_ext.get_projections_one_unit(unit_id) 
    except Exception as e:
        print(f"\n[ERROR] No se encontraron datos para la unidad '{unit_id}'. Detalles: {e}")
        return
        
    if wfs is None or len(wfs) == 0:
        print(f"\n[ERROR] No se encontraron espigas para la unidad '{unit_id}'.")
        return

    # 2. Preparación de datos
    promedio_global = np.mean(wfs, axis=0)
    best_channel_idx = np.argmin(np.min(promedio_global, axis=0))
    X = pcs.reshape(pcs.shape[0], -1) 

    # 3. Configurar la cuadrícula de figuras (3 filas x 3 columnas para K de 2 a 10)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    # 4. Iterar de K=2 hasta K=10
    for k in range(2, max_clusters + 1):
        ax = axes[k - 2] # El índice 0 es K=2
        
        # Machine Learning
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Evaluar la calidad de la separación con Silhouette Score (Más cerca de 1.0 es mejor)
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = 0.0
        
        # Graficar cada sub-clúster
        for cluster_label in range(k):
            espigas_del_grupo = wfs[labels == cluster_label, :, best_channel_idx]
            
            # Solo graficamos si el grupo no quedó vacío
            if len(espigas_del_grupo) > 0:
                template_grupo = np.mean(espigas_del_grupo, axis=0)
                ax.plot(template_grupo, label=f'C{cluster_label} (n={len(espigas_del_grupo)})', linewidth=2)
            
        ax.set_title(f'Prueba K={k} | Silueta: {sil_score:.3f}')
        if k >= 8: # Solo poner etiquetas de X en la última fila
            ax.set_xlabel('Muestras de tiempo')
        if k in [2, 5, 8]: # Solo poner etiquetas de Y en la primera columna
            ax.set_ylabel('Amplitud (V)')
            
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Evolución K-Means: {unit_id}\n(Busca el gráfico con el valor de "Silueta" más alto)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =========================================================
# MAIN (CARGA AUTOMÁTICA DE LA PRIMERA UNIDAD)
# =========================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    # 1. Seleccionar la carpeta del Analizador
    selected_folder_path = filedialog.askdirectory(
        title="Selecciona la carpeta 'unified_analyzer'",
        mustexist=True
    )

    if not selected_folder_path:
        print("Operación cancelada.")
        exit()

    print(f"Cargando analizador desde: {selected_folder_path}")
    
    try:
        analyzer = sc.load_sorting_analyzer(selected_folder_path)
    except Exception as e:
        print(f"\n[ERROR] No se pudo cargar el analizador: {e}")
        exit()

    # 2. Obtener lista de IDs y tomar SOLO EL PRIMERO automáticamente
    unidades_validas = list(analyzer.sorting.get_unit_ids())
    
    if not unidades_validas:
        print("\n[ERROR] No hay unidades detectadas en este analizador.")
        exit()
        
    unidad_a_evaluar = unidades_validas[0]
    
    print(f"\n=== SISTEMATIZACIÓN K-MEANS ===")
    print(f"Total de unidades en la base de datos: {len(unidades_validas)}")
    print(f"Seleccionando automáticamente la unidad: {unidad_a_evaluar}")

    # 3. Ejecutar el algoritmo
    explorar_splits_kmeans_sistematico(analyzer, unidad_a_evaluar, max_clusters=10)