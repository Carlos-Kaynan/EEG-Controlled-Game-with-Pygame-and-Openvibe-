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





# outro exemplo


'''
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter
import numpy as np
import time

#  Configurações
SAMPLING_RATE = 512  # Hz
WINDOW_DURATION = 1  # segundos
WINDOW_SIZE = SAMPLING_RATE * WINDOW_DURATION

# Índices dos canais (assumindo ordem conhecida)
CHANNELS = {
    'C3': 0,
    'C4': 1,
    'Pz': 2,
    'Fz': 3
}

# Limiar de potência por banda
THRESHOLDS = {
    'beta': 8.0,
    'alpha': 10.0,
    'theta': 7.0
}

#  Filtro passa-faixa
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

#  Classificação
def classificar(window):
    sinais = {}

    for canal_nome, idx in CHANNELS.items():
        sinal = np.array(window)[:, idx]

        sinais[canal_nome] = {
            'beta': np.mean(bandpass_filter(sinal, 13, 30, SAMPLING_RATE) ** 2),
            'alpha': np.mean(bandpass_filter(sinal, 8, 13, SAMPLING_RATE) ** 2),
            'theta': np.mean(bandpass_filter(sinal, 4, 8, SAMPLING_RATE) ** 2),
        }

    # Regras
    if sinais['C3']['beta'] > THRESHOLDS['beta'] or sinais['C4']['beta'] > THRESHOLDS['beta']:
        print(0)
    elif sinais['Pz']['alpha'] > THRESHOLDS['alpha']:
        print(1)
    elif sinais['Fz']['theta'] > THRESHOLDS['theta']:
        print(2)
    else:
        print("Nenhuma banda dominante detectada.")

#  Recebendo sinais do OpenViBE
print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')  
inlet = StreamInlet(streams[0])

print("Recebendo dados EEG...")

window = []

while True:
    sample, timestamp = inlet.pull_sample()
    if sample:
        window.append(sample)

        # Quando tiver uma janela completa
        if len(window) >= WINDOW_SIZE:
            classificar(window)
            window = []  # Reinicia a janela

'''
