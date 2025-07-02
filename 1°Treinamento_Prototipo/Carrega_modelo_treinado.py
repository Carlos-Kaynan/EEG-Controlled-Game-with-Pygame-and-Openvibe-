# realtime_motor_classifier.py
# Usa o modelo SVM para classificar sinais EEG em tempo real: abrir, fechar ou neutro

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from scipy.signal import butter, lfilter, welch
import joblib

# === Carregar modelo treinado ===
clf = joblib.load("C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\svm_motor_imagery.joblib")

# === Filtros ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

def extract_features(window, fs):
    c3 = window[:, 0]
    c4 = window[:, 1]

    mu_c3 = bandpass_filter(c3, 8, 13, fs)
    beta_c3 = bandpass_filter(c3, 13, 30, fs)
    mu_c4 = bandpass_filter(c4, 8, 13, fs)
    beta_c4 = bandpass_filter(c4, 13, 30, fs)

    def band_power(signal):
        freqs, psd = welch(signal, fs)
        return np.sum(psd)

    return [
        band_power(mu_c3),
        band_power(beta_c3),
        band_power(mu_c4),
        band_power(beta_c4)
    ]

# === Parâmetros ===
fs = 512
window_size = 1  # segundos
samples_per_window = int(fs * window_size)

print("Procurando stream EEG...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream encontrado.")
    exit()

inlet = StreamInlet(streams[0])
info = inlet.info()
desc = info.desc()

# Detectar nomes dos canais
ch = desc.child('channels').child('channel')
channel_names = []
for i in range(info.channel_count()):
    label = ch.child_value('label')
    channel_names.append(label)
    ch = ch.next_sibling()

print("Canais detectados:", channel_names)

try:
    idx_c3 = channel_names.index('10') #ver se o indice é o 10 ou o valor da coluna "11"
    idx_c4 = channel_names.index('13') #ver se o indice é o 13 ou o valor da coluna "14"
except ValueError:
    print("Erro: Canais C3 e/ou C4 não encontrados no stream EEG.")
    exit()

print(f"Usando C3 (canal {idx_c3}), C4 (canal {idx_c4})")

buffer = []
print("\nClassificando em tempo real (pressione CTRL+C para parar)...")
try:
    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample)

        if len(buffer) >= samples_per_window:
            window = np.array(buffer[-samples_per_window:])
            c3 = window[:, idx_c3]
            c4 = window[:, idx_c4]
            features = extract_features(np.column_stack((c3, c4)), fs)
            pred = clf.predict([features])[0]

            if pred == 1:
                print("🖐 Abrir Mão")
            elif pred == 2:
                print("✊ Fechar Mão")
            else:
                print("⏸ Neutro")

except KeyboardInterrupt:
    print("\nClassificação encerrada.")
