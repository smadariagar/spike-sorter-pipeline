import os
import json
import numpy as np
import probeinterface as pi

# Parámetros que usaste
num_channels = 60
pitch = 200
radius = 15
export_folder = "/home/samuel/Documentos/Explora/spike_sorter/data/MEA33/tdc_ready_test"

# 1. Recrear la sonda (usando la lógica de h5)
json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mea_mapping.json')
with open(json_path, 'r') as f:
    mea_mapping = json.load(f)

list_2_map = mea_mapping["channel_mapping_h5"]

probe = pi.Probe(ndim=2, si_units='um')
positions, valid_channel_indices = [], []

for i, num in enumerate(list_2_map):
    num_str = str(num)
    if num_str == '0':
        continue
    x = (int(num_str[0]) - 1) * pitch
    y = (8 - int(num_str[1])) * pitch
    positions.append([x, y])
    valid_channel_indices.append(i)

probe.set_contacts(positions=np.array(positions), shapes='circle', shape_params={'radius': radius})
probe.set_device_channel_indices(valid_channel_indices)

# 2. EL ARREGLO: Envolver en ProbeGroup y guardar
probegroup = pi.ProbeGroup()
probegroup.add_probe(probe)

prb_path = os.path.join(export_folder, "mea_probe.prb")
pi.write_prb(prb_path, probegroup)

print(f"¡Rescate exitoso! Archivo guardado en: {prb_path}")