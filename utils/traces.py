import spikeinterface.core as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt

# Ruta exacta según la terminal que me mostraste antes
folder_path = "/home/samuel/Documentos/Explora/spike_sorter/data/MEA36/tdc_ready_ses_1" 

# Usamos 'load' en lugar de 'load_extractor' para las versiones nuevas
recording = sc.load(folder_path)

print("Generando gráfico...")
# Mostramos el primer segundo de grabación para los primeros 10 canales
w = sw.plot_traces(recording, time_range=(0, 1), channel_ids=recording.get_channel_ids()[0:10])

plt.show()