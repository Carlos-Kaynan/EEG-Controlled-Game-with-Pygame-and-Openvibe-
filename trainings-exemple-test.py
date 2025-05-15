import mne
from mne.datasets import eegbci
from mne.io import read_raw_gdf
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar o arquivo GDF exportado do OpenViBE
raw = read_raw_gdf("sujeito01_motorimagery.gdf", preload=True)

# Filtrar (opcional)
raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')

# Eventos e anotações
events, event_id = mne.events_from_annotations(raw)

# Mapear eventos de imaginação (ajuste conforme seu cenário OpenViBE)
label_map = {
    'T0': 0,  # repouso (opcional)
    'T1': 1,  # imaginação de fechar
    'T2': 2   # imaginação de abrir
}
epochs = mne.Epochs(raw, events, event_id=label_map,
                    tmin=0.0, tmax=4.0, baseline=None, preload=True)

X = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
y = epochs.events[:, -1]
