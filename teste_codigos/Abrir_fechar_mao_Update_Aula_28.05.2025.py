
'''
from pylsl import StreamInlet, resolve_byprop
from pylsl import StreamInlet, resolve_byprop
import time

print("Procurando rede EEG LSL")
streams = resolve_byprop('name','openvibeSignal')

inlet = StreamInlet(streams[0])

eeg_lista = []
print("Recebendo EEG")
try:
    start_time  = time.time()
    while time.time() - start_time < 20:
        sample, timestamp = inlet.pull_sample()
        eeg_lista.append(sample)
        print(f"Timestamp: {timestamp}, Dados: {sample}")

except KeyboardInterrupt:
    print("Interrupção Manual.")
finally:
    print("coleta finalizada")
    print(f"Total de amostras coletadas: {len(eeg_lista)}")

'''





#fazer experimento com exoesqueleto

from pylsl import StreamInlet, resolve_byprop
import time
import numpy as np
from scipy.signal import butter, lfilter, welch

# Funções auxiliares para filtragem
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

# Parâmetros
fs = 512  # Frequência de amostragem em Hz (ajuste conforme sua aquisição)
window_size = 1  # segundos
samples_per_window = int(fs * window_size)

print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream encontrado.")
    exit()

inlet = StreamInlet(streams[0])
print("Coletando e classificando sinais EEG...")

try:
    buffer = []
    start_time = time.time()
    while time.time() - start_time < 60:  # rodar por 20s

        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample)

        if len(buffer) >= samples_per_window:
            window = np.array(buffer[-samples_per_window:])  # janela mais recente
            window = np.array(window)

            # Supondo que os canais C3 e C4 estão nas posições 0 e 1
            c3 = window[:, 0]
            c4 = window[:, 1]

            # Filtrar bandas mu (8–13 Hz) e beta (13–30 Hz)
            mu_c3 = bandpass_filter(c3, 8, 13, fs)
            beta_c3 = bandpass_filter(c3, 13, 30, fs)

            mu_c4 = bandpass_filter(c4, 8, 13, fs)
            beta_c4 = bandpass_filter(c4, 13, 30, fs)

            # Extrair potência (usando PSD - Power Spectral Density)
            def band_power(signal):
                freqs, psd = welch(signal, fs)
                return np.sum(psd)

            #mu_power_diff = band_power(mu_c4) - band_power(mu_c3)
            #beta_power_diff = band_power(beta_c4) - band_power(beta_c3)

            mu_power_diff = band_power(mu_c4) / band_power(beta_c4)
            beta_power_diff = band_power(mu_c3) / band_power(beta_c3)

            print(f' Mu:{mu_power_diff} e  Beta: {beta_power_diff}')
                  

            # Classificação simulada:
            # Se atividade aumentar mais em C4, imagina abrir (mão direita)
            # Se em C3, imagina fechar (mão esquerda)
            if mu_power_diff > 0.5 or beta_power_diff > 0.5:
                pass
                #print(" Abrir mão")
            elif mu_power_diff < 0.5 or beta_power_diff < 0.5:
                pass
                #print(" Fechar mão")
            else:
                pass
                #print("⏸ Neutro / sem decisão")

except KeyboardInterrupt:
    print("Interrompido.")
finally:
    print("Encerrando classificação.")
