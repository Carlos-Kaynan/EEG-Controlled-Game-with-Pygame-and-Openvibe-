import numpy as np
from pylsl import StreamInlet, resolve_streams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pygame
import time
from pylsl import resolve_byprop

# ===============================
# 1. CAPTURA EEG (via LSL)
# ===============================
print("Procurando stream EEG...")
streams = resolve_byprop('name', 'openvibeSignal')
inlet = StreamInlet(streams[0])

# Função para capturar uma janela de EEG
def get_eeg_window(window_size=3, fs=250):
    """Coleta janela de EEG em segundos"""
    samples = []
    for _ in range(window_size * fs):
        sample, _ = inlet.pull_sample()
        samples.append(sample)
    return np.array(samples)

# ===============================
# 2. FEATURES
# ===============================
def extract_features(eeg_window):
    """Extrai features simples (média por canal)"""
    features = np.mean(eeg_window, axis=0)
    return features

# ===============================
# 3. TREINAMENTO
# ===============================
print("Coletando dados de treino...")

X, y = [], []

# 20 trials de 3 segundos cada
n_trials = 20
window_size = 3

for label, comando in enumerate(["esquerda", "direita"]):
    print(f"Prepare-se para pensar em {comando} ({n_trials} vezes, {window_size}s cada)")
    time.sleep(3)

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials} - Pense em {comando} agora!")
        eeg = get_eeg_window(window_size=window_size)  # 3 segundos de EEG
        feat = extract_features(eeg)
        X.append(feat)
        y.append(label)
        print("OK, pode relaxar 2s...")
        time.sleep(2)

X = np.array(X)
y = np.array(y)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print("Treinamento concluído!")

# ===============================
# 4. INTEGRAÇÃO COM JOGO (pygame)
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

    # Coleta 3 segundos de EEG para predição
    eeg = get_eeg_window(window_size=3)
    feat = extract_features(eeg)
    prediction = clf.predict([feat])[0]

    if prediction == 0:  # esquerda
        player.x -= 10
    elif prediction == 1:  # direita
        player.x += 10

    # Limites da tela
    player.x = max(0, min(550, player.x))

    # Renderização
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (0, 255, 0), player)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()