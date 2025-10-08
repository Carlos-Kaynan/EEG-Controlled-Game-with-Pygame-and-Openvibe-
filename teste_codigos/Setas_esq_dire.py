import pygame
import time
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
from pylsl import resolve_byprop


# === Inicializar LSL (EEG vindo do OpenViBE) ===
print("Procurando fluxo EEG...")
streams = resolve_byprop('name', 'openvibeSignal')
inlet = StreamInlet(streams[0])
fs = 128  # taxa de amostragem do OpenViBE (ajuste conforme seu cenário)
window_size = 2  # segundos de coleta por estímulo
n_samples = int(fs * window_size)

# === CONFIGURAÇÕES DO EEG ===
fs = 128         # taxa de amostragem (ajuste conforme OpenViBE)
window_size = 2  # segundos de sinal coletado após estímulo
n_samples = int(fs * window_size)
n_channels = 32  # número de canais EEG
duration = 60    # segundos de duração por direção

# === CONFIGURAÇÕES DO PYGAME ===
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Coleta EEG - Estímulos")
font = pygame.font.Font(None, 120)
clock = pygame.time.Clock()

def draw_arrow(direction):
    """Desenha seta na tela"""
    screen.fill((0, 0, 0))
    arrow = "<" if direction == "left" else ">"
    color = (0, 255, 0) if direction == "left" else (0, 128, 255)
    text = font.render(arrow, True, color)
    rect = text.get_rect(center=(300, 200))
    screen.blit(text, rect)
    pygame.display.flip()

def collect_data(direction):
    """Coleta sinais EEG para uma direção específica"""
    print(f"\n=== Iniciando coleta para {direction.upper()} ===")
    X = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Exibir seta
        draw_arrow(direction)
        stimulus_time = time.time()

        # Esperar 2 segundos (tempo para imaginar movimento)
        while time.time() - stimulus_time < 2:
            pygame.event.pump()
            clock.tick(30)

        # Coletar janela EEG (2 segundos)
        data_window = []
        for _ in range(n_samples):
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample is not None:
                data_window.append(sample)
            pygame.event.pump()
            clock.tick(60)

        if len(data_window) == n_samples:
            X.append(np.array(data_window))
            print(f"✓ Janela coletada ({len(X)} janelas até agora)")
        else:
            print("⚠️  Janela incompleta, ignorada.")

        # Intervalo entre tentativas
        interval_start = time.time()
        screen.fill((0, 0, 0))
        pygame.display.flip()
        while time.time() - interval_start < 1:
            pygame.event.pump()
            clock.tick(30)

    # Salvar dados em CSV
    X = np.array(X)
    df = pd.DataFrame(X.reshape(X.shape[0], -1))
    nome_arquivo = f"dados_{direction}.csv"
    df.to_csv(nome_arquivo, index=False)
    print(f"✅ Dados salvos em {nome_arquivo} ({df.shape})")

# === COLETA PARA ESQUERDA E DIREITA ===
try:
    collect_data("left")
    collect_data("right")
    pygame.quit()
    print("\nColeta concluída com sucesso!")

except KeyboardInterrupt:
    pygame.quit()
    print("\nColeta interrompida pelo usuário.")




    