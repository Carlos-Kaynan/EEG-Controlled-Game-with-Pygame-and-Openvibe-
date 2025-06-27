
#original
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




""""
/*#fazer experimento com exoesqueleto

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
    while time.time() - start_time < 600:  # rodar por 20s

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

            #print(f' Mu:{mu_power_diff} e  Beta: {beta_power_diff}')
                  

            # Classificação simulada:
            # Se atividade aumentar mais em C4, imagina abrir (mão direita)
            # Se em C3, imagina fechar (mão esquerda)
            if mu_power_diff > 0.5 or beta_power_diff > 0.5:
                pass
                print(" Abrir mão")
            elif mu_power_diff < 0.5 or beta_power_diff < 0.5:
                pass
                print(" Fechar mão")
            else:
                pass
                print("⏸ Neutro / sem decisão")

except KeyboardInterrupt:
    print("Interrompido.")
finally:
    print("Encerrando classificação.")
"""
#codigo modificado pelo chat gpt 1
"""
from pylsl import StreamInlet, resolve_byprop
import time
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from collections import deque

# ------------------------------------
# Funções auxiliares
# ------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

# ------------------------------------
# Parâmetros
# ------------------------------------
fs = 512  # frequência de amostragem
window_size = 1  # segundos
samples_per_window = int(fs * window_size)
plot_duration = 5  # segundos para manter no gráfico
max_points = fs * plot_duration

limiar_mu = 1e-6
limiar_beta = 1e-6

# ------------------------------------
# Conectar ao stream LSL
# ------------------------------------
print("Procurando stream EEG LSL...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream EEG encontrado.")
    exit()

inlet = StreamInlet(streams[0])
print("Stream EEG conectado. Coletando dados...")

# ------------------------------------
# Inicializar visualização
# ------------------------------------
plt.ion()
fig, axs = plt.subplots(8, 4, figsize=(15, 10))
axs = axs.flatten()

channel_buffers = [deque(maxlen=max_points) for _ in range(32)]
time_buffer = deque(maxlen=max_points)

lines = []
x = np.linspace(-plot_duration, 0, max_points)
for i in range(32):
    line, = axs[i].plot(x, np.zeros_like(x))
    axs[i].set_title(f'Canal {i+1}')
    axs[i].set_ylim(-100, 100)
    axs[i].set_xlim(-plot_duration, 0)
    lines.append(line)

# ------------------------------------
# Loop principal
# ------------------------------------
buffer = []
start_time = time.time()

try:
    while time.time() - start_time < 600:  # 10 minutos

        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample)
            sample_array = np.array(sample)
            current_time = time.time() - start_time
            time_buffer.append(current_time)

            for ch in range(32):
                channel_buffers[ch].append(sample_array[ch])

        # Quando tiver uma janela cheia
        if len(buffer) >= samples_per_window:
            window = np.array(buffer[-samples_per_window:])

            # Supondo C3 = canal 1, C4 = canal 2
            c3 = window[:, 0]
            c4 = window[:, 1]

            mu_c3 = bandpass_filter(c3, 8, 13, fs)
            beta_c3 = bandpass_filter(c3, 13, 30, fs)

            mu_c4 = bandpass_filter(c4, 8, 13, fs)
            beta_c4 = bandpass_filter(c4, 13, 30, fs)

            mu_diff = band_power(mu_c4, fs) - band_power(mu_c3, fs)
            beta_diff = band_power(beta_c4, fs) - band_power(beta_c3, fs)

            # Classificação
            if mu_diff > limiar_mu or beta_diff > limiar_beta:
                print("🖐 Abrir mão")
            elif mu_diff < -limiar_mu or beta_diff < -limiar_beta:
                print("✊ Fechar mão")
            else:
                print("⏸ Neutro")

            # Atualizar gráfico (a cada janela)
            for i in range(32):
                y = np.array(channel_buffers[i])
                if len(y) < max_points:
                    y = np.pad(y, (max_points - len(y), 0), 'constant')
                lines[i].set_ydata(y)
            x_vals = np.linspace(-plot_duration, 0, max_points)
            for line in lines:
                line.set_xdata(x_vals)
            plt.pause(0.01)

except KeyboardInterrupt:
    print("🛑 Interrompido manualmente.")
finally:
    print("✅ Encerrando experimento.")
    np.save("dados_eeg.npy", np.array(buffer))"""
"""
import sys
import numpy as np
from scipy.signal import butter, filtfilt, welch, sosfiltfilt
from pylsl import StreamInlet, resolve_byprop
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# --- Funções de filtro ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, data)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

# --- Parâmetros ---
fs = 512
win_sec = 1
win_samps = fs*win_sec
plot_sec = 5
buf_len = fs*plot_sec

lim_mu, lim_beta = 1e-6, 1e-6

# --- Conexão LSL ---
print("Procurando stream EEG LSL...")
streams = resolve_byprop('name','openvibeSignal')
if not streams:
    print("Nenhum stream EEG encontrado.")
    sys.exit()

inlet = StreamInlet(streams[0])
print("Stream conectado.")

# --- Buffer circular 32×buf_len ---
buf = np.zeros((32, buf_len))
time_buf = np.linspace(-plot_sec, 0, buf_len)

# --- Setup gráfico PyQtGraph ---
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG - 32 canais")
win.resize(1200,800)
plots, curves = [], []
for i in range(32):
    p = win.addPlot(row=i//4, col=i%4, title=f"Canal {i+1}")
    p.setYRange(-100,100)
    curve = p.plot(pen='w', width=1)
    plots.append(p); curves.append(curve)

# --- Timer de atualização ---
def update():
    global buf
    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample:
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = sample

    # Processa por janela completa
    if update.counter % win_samps == 0:
        c3 = buf[0, -win_samps:]
        c4 = buf[1, -win_samps:]
        mu_diff = band_power(bandpass_filter(c4,8,13,fs),fs) - band_power(bandpass_filter(c3,8,13,fs),fs)
        beta_diff = band_power(bandpass_filter(c4,13,30,fs),fs) - band_power(bandpass_filter(c3,13,30,fs),fs)
        if mu_diff > lim_mu or beta_diff > lim_beta:
            print("🖐 Abrir mão")
        elif mu_diff < -lim_mu or beta_diff < -lim_beta:
            print("✊ Fechar mão")
        else:
            print("⏸ Neutro")

        # Atualiza gráficos
        for i, curve in enumerate(curves):
            curve.setData(time_buf, buf[i])
    update.counter += 1

update.counter = 0
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000/fs))  # executa a cada ~2 ms

# --- Execução ---
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
"""
"""
import sys
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from pylsl import StreamInlet, resolve_byprop
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# --- Funções auxiliares ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, data)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

# --- Parâmetros ---
fs = 512
win_sec = 1
win_samps = fs * win_sec
plot_sec = 5
buf_len = fs * plot_sec
lim_mu, lim_beta = 1e-6, 1e-6

# --- Conectar ao stream LSL ---
print("Procurando stream EEG LSL...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream EEG encontrado.")
    sys.exit()

inlet = StreamInlet(streams[0])
info = inlet.info()
n_channels = info.channel_count()

# Obter nomes dos canais
desc = info.desc()
channels = desc.child('channels').child('channel')
ch_names = []
for i in range(n_channels):
    ch_names.append(channels.child_value('label'))
    channels = channels.next_sibling()

print(f"Stream conectado com {n_channels} canais:")
print("Nomes dos canais:", ch_names)

# Identificar índices dos canais C3 e C4
try:
    c3_index = ch_names.index('C3')
    c4_index = ch_names.index('C4')
except ValueError:
    print("Canais C3 ou C4 não encontrados no stream!")
    sys.exit()

# --- Buffer ---
buf = np.zeros((n_channels, buf_len))
time_buf = np.linspace(-plot_sec, 0, buf_len)

# --- Setup gráfico ---
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG - Visualização em Tempo Real")
win.resize(1600, 900)
plots, curves = [], []

for i in range(n_channels):
    p = win.addPlot(row=i//4, col=i%4, title=f"Canal {ch_names[i]}")
    p.setYRange(-100, 100)
    curve = p.plot(pen='w', width=1)
    plots.append(p)
    curves.append(curve)

# --- Loop de atualização ---
def update():
    global buf
    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample:
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = sample[:n_channels]

    # Processamento por janela
    if update.counter % win_samps == 0:
        c3 = buf[c3_index, -win_samps:]
        c4 = buf[c4_index, -win_samps:]

        mu_diff = band_power(bandpass_filter(c4, 8, 13, fs), fs) - band_power(bandpass_filter(c3, 8, 13, fs), fs)
        beta_diff = band_power(bandpass_filter(c4, 13, 30, fs), fs) - band_power(bandpass_filter(c3, 13, 30, fs), fs)

        if mu_diff > lim_mu or beta_diff > lim_beta:
            print("🖐 Abrir mão")
        elif mu_diff < -lim_mu or beta_diff < -lim_beta:
            print("✊ Fechar mão")
        else:
            print("⏸ Neutro")

        # Atualiza gráficos
        for i, curve in enumerate(curves):
            curve.setData(time_buf, buf[i])

    update.counter += 1

update.counter = 0
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000/fs))  # Atualiza a cada 2ms (512Hz)

# --- Execução ---
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
"""

import sys
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from pylsl import StreamInlet, resolve_byprop
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

e=0
d=0

# --- Funções auxiliares ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, data)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

# --- Parâmetros ---
fs = 512
win_sec = 1
win_samps = fs * win_sec
plot_sec = 5
buf_len = fs * plot_sec
#lim_mu, lim_beta = 1e-6, 1e-6
lim_mu, lim_beta = 0.75, 0.75
# --- Conectar ao stream LSL ---
print("Procurando stream EEG LSL...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream EEG encontrado.")
    sys.exit()

inlet = StreamInlet(streams[0])
info = inlet.info()
n_channels = info.channel_count()

# Obter nomes dos canais
desc = info.desc()
channels = desc.child('channels').child('channel')
ch_names = []
for i in range(n_channels):
    ch_names.append(channels.child_value('label'))
    channels = channels.next_sibling()

print(f"Stream conectado com {n_channels} canais:")
print("Nomes dos canais:", ch_names)

# Selecionar canais '11' e '14'
try:
    ch1_index = ch_names.index('11')
    ch2_index = ch_names.index('14')
except ValueError:
    print("Canais '11' e '14' não encontrados no stream!")
    sys.exit()

# --- Buffer ---
buf = np.zeros((n_channels, buf_len))
time_buf = np.linspace(-plot_sec, 0, buf_len)

# --- Setup gráfico ---
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG - Visualização em Tempo Real")
win.resize(1600, 900)
plots, curves = [], []

for i in range(n_channels):
    p = win.addPlot(row=i//4, col=i%4, title=f"Canal {ch_names[i]}")
    p.setYRange(-100, 100)
    curve = p.plot(pen='w', width=1)
    plots.append(p)
    curves.append(curve)

# --- Loop de atualização ---
def update():
    global buf, d, e
    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample:
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = sample[:n_channels]

    # Processamento por janela
    if update.counter % win_samps == 0:
        ch1 = buf[ch1_index, -win_samps:]
        ch2 = buf[ch2_index, -win_samps:]

        mu_diff = band_power(bandpass_filter(ch2, 8, 13, fs), fs) - band_power(bandpass_filter(ch1, 8, 13, fs), fs)
        #beta_diff = band_power(bandpass_filter(ch2, 13, 30, fs), fs) - band_power(bandpass_filter(ch1, 13, 30, fs), fs)
        
        if mu_diff > lim_mu: #or beta_diff > lim_beta
            print("🖐 Esquerda")
            e+=1
            print("Esquerda:"+str(e)+"\nDireita:"+str(d))
        elif mu_diff < -lim_mu: #or beta_diff < -lim_beta
            print("✊ Direita")
            d+=1
            print("Esquerda:"+str(e)+"\nDireita:"+str(d))
        else:
            print("⏸ Neutro")
        # Atualiza gráficos
        for i, curve in enumerate(curves):
            curve.setData(time_buf, buf[i])

    update.counter += 1

update.counter = 0
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000/fs))  # Atualiza a cada 2ms (512Hz)

# --- Execução ---
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
