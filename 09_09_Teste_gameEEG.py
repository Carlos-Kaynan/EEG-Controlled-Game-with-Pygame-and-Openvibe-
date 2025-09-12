import numpy as np
from pylsl import StreamInlet, resolve_streams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pygame
import time
from pylsl import resolve_byprop
from scipy.signal import butter, lfilter, iirnotch, welch


#Treinamento com filtro de Ruidos eletricos, piscada e expasmo muscular


# ===============================
# 1. CAPTURA EEG (via LSL)
# ===============================
print("Procurando stream EEG...")
streams = resolve_byprop('name', 'openvibeSignal')  # procura o stream EEG do Nautilus
inlet = StreamInlet(streams[0])

# Info do stream
info = inlet.info()
n_channels = info.channel_count()
print(f"Número de canais: {n_channels}\n")

# Lista canais
labels = []
ch = info.desc().child("channels").first_child()
for i in range(n_channels):
    label = ch.child_value("label")
    labels.append(label)
    print(f"Canal {i} → {label}")
    ch = ch.next_sibling()

# Seleciona C3, Cz, C4 se disponíveis
canais_motor = []
for alvo in ["C3", "Cz", "C4"]:
    if alvo in labels:
        idx = labels.index(alvo)
        canais_motor.append(idx)
        print(f"Usando canal {alvo} (índice {idx})")

if not canais_motor:
    print("⚠️ Nenhum dos canais C3, Cz, C4 encontrado! Usando todos como fallback.")
    canais_motor = list(range(n_channels))

# ===============================
# 2. FILTROS
# ===============================
fs = 250  # frequência de amostragem típica do Nautilus (ajuste se necessário)

# Filtro notch (remove 60Hz da rede elétrica)
b_notch, a_notch = iirnotch(w0=60, Q=30, fs=fs)

# Filtro passa-banda (8–30Hz)
b_band, a_band = butter(4, [8, 30], btype='band', fs=fs)

def preprocess(eeg):
    """Aplica notch + passa-banda em cada canal"""
    eeg = lfilter(b_notch, a_notch, eeg, axis=0)
    eeg = lfilter(b_band, a_band, eeg, axis=0)
    return eeg

# ===============================
# 3. EXTRAÇÃO DE FEATURES
# ===============================
def get_eeg_window(window_size=3, fs=250):
    """Coleta janela de EEG em segundos"""
    samples = []
    for _ in range(window_size * fs):
        sample, _ = inlet.pull_sample()
        sample_motor = [sample[i] for i in canais_motor]
        samples.append(sample_motor)
    return np.array(samples)

def extract_features(eeg_window):
    """Extrai potência média nas bandas mu (8–13Hz) e beta (14–30Hz)"""
    eeg_filt = preprocess(eeg_window)
    feats = []
    for ch in range(eeg_filt.shape[1]):
        f, Pxx = welch(eeg_filt[:, ch], fs=fs, nperseg=fs*2)
        # Potência nas bandas
        mu_power = np.mean(Pxx[(f >= 8) & (f <= 13)])
        beta_power = np.mean(Pxx[(f >= 14) & (f <= 30)])
        feats.extend([mu_power, beta_power])
    return np.array(feats)

# ===============================
# 4. TREINAMENTO
# ===============================
print("\n=== FASE DE TREINAMENTO ===")
X, y = [], []
n_trials = 100        # aumente para 100+ em treino real
window_size = 3      # segundos por trial

for label, comando in enumerate(["esquerda", "direita"]):
    print(f"\nPrepare-se para pensar em {comando} ({n_trials} vezes, {window_size}s cada)")
    time.sleep(3)

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials} - Pense em {comando} agora!")
        eeg = get_eeg_window(window_size=window_size)
        feat = extract_features(eeg)
        X.append(feat)
        y.append(label)
        print("OK, pode relaxar 2s...")
        time.sleep(2)

X = np.array(X)
y = np.array(y)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print("\nTreinamento concluído! Iniciando o jogo...")

# ===============================
# 5. JOGO COM PYGAME
# ===============================
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("EEG Control Game")
clock = pygame.time.Clock()

player = pygame.Rect(300, 350, 50, 50)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Coleta janela e faz predição
    eeg = get_eeg_window(window_size=3)
    feat = extract_features(eeg)
    prediction = clf.predict([feat])[0]

    if prediction == 0:
        player.x -= 10
        print("← Movimento detectado: esquerda")
    elif prediction == 1:
        player.x += 10
        print("→ Movimento detectado: direita")

    # Limites da tela
    player.x = max(0, min(550, player.x))

    # Renderização
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (0, 255, 0), player)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()




























#Edição 10/09/2025

import numpy as np
from pylsl import StreamInlet, resolve_streams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pygame
import time
from pylsl import resolve_byprop
from scipy.signal import butter, lfilter, iirnotch, welch


#Treinamento com filtro de Ruidos eletricos, piscada e expasmo muscular


# ===============================
# 1. CAPTURA EEG (via LSL)
# ===============================
print("Procurando stream EEG...")
streams = resolve_byprop('name', 'openvibeSignal')  # procura o stream EEG do Nautilus
inlet = StreamInlet(streams[0])

# Info do stream
info = inlet.info()
n_channels = info.channel_count()
print(f"Número de canais: {n_channels}\n")

# Lista canais
labels = []
ch = info.desc().child("channels").first_child()
for i in range(n_channels):
    label = ch.child_value("label")
    labels.append(label)
    print(f"Canal {i} → {label}")
    ch = ch.next_sibling()

# Seleciona C3, Cz, C4 se disponíveis
canais_motor = []
for alvo in ["C3", "Cz", "C4"]:
    if alvo in labels:
        idx = labels.index(alvo)
        canais_motor.append(idx)
        print(f"Usando canal {alvo} (índice {idx})")

if not canais_motor:
    print("⚠️ Nenhum dos canais C3, Cz, C4 encontrado! Usando todos como fallback.")
    canais_motor = list(range(n_channels))

# ===============================
# 2. FILTROS
# ===============================
fs = 250  # frequência de amostragem típica do Nautilus (ajuste se necessário)

# Filtro notch (remove 60Hz da rede elétrica)
b_notch, a_notch = iirnotch(w0=60, Q=30, fs=fs)

# Filtro passa-banda (8–30Hz)
b_band, a_band = butter(4, [8, 30], btype='band', fs=fs)

def preprocess(eeg):
    """Aplica notch + passa-banda em cada canal"""
    eeg = lfilter(b_notch, a_notch, eeg, axis=0)
    eeg = lfilter(b_band, a_band, eeg, axis=0)
    return eeg

# ===============================
# 3. EXTRAÇÃO DE FEATURES
# ===============================
def get_eeg_window(window_size=3, fs=250):
    """Coleta janela de EEG em segundos"""
    samples = []
    for _ in range(window_size * fs):
        sample, _ = inlet.pull_sample()
        sample_motor = [sample[i] for i in canais_motor]
        samples.append(sample_motor)
    return np.array(samples)

def extract_features(eeg_window):
    """Extrai potência média nas bandas mu (8–13Hz) e beta (14–30Hz)"""
    eeg_filt = preprocess(eeg_window)
    feats = []
    for ch in range(eeg_filt.shape[1]):
        f, Pxx = welch(eeg_filt[:, ch], fs=fs, nperseg=fs*2)
        # Potência nas bandas
        mu_power = np.mean(Pxx[(f >= 8) & (f <= 13)])
        beta_power = np.mean(Pxx[(f >= 14) & (f <= 30)])
        feats.extend([mu_power, beta_power])
    return np.array(feats)

# ===============================
# 4. TREINAMENTO
# ===============================
print("\n=== FASE DE TREINAMENTO ===")
X, y = [], []
n_trials = 20        # aumente para 100+ em treino real
window_size = 3      # segundos por trial

for label, comando in enumerate(["esquerda", "direita"]):
    print(f"\nPrepare-se para pensar em {comando} ({n_trials} vezes, {window_size}s cada)")
    time.sleep(3)

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials} - Pense em {comando} agora!")
        eeg = get_eeg_window(window_size=window_size)
        feat = extract_features(eeg)
        X.append(feat)
        y.append(label)
        print("OK, pode relaxar 2s...")
        time.sleep(2)

X = np.array(X)
y = np.array(y)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print("\nTreinamento concluído! Iniciando o jogo...")

# ===============================
# 5. JOGO COM PYGAME
# ===============================
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("EEG Control Game")
clock = pygame.time.Clock()

player = pygame.Rect(300, 350, 50, 50)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Coleta janela e faz predição
    eeg = get_eeg_window(window_size=3)
    feat = extract_features(eeg)
    prediction = clf.predict([feat])[0]

    if prediction == 0:
        player.x -= 10
        print("← Movimento detectado: esquerda")
    elif prediction == 1:
        player.x += 10
        print("→ Movimento detectado: direita")

    # Limites da tela
    player.x = max(0, min(550, player.x))

    # Renderização
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (0, 255, 0), player)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
