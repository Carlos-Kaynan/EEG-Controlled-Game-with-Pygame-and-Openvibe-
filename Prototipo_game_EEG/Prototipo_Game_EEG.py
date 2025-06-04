
'''
import pygame
import sys
import time

# Inicializar o Pygame
pygame.init()

# Constantes
LARGURA, ALTURA = 800, 400
FPS = 60
TAMANHO_PERSONAGEM = (50, 100)
GRAVIDADE = 1
PULO = -20

# Cores
BRANCO = (255, 255, 255)

# Criar tela
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Jogo com 3 Fases")

# Carregar backgrounds
backgrounds = [
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\cidade.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\floresta.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\OneDrive\\Área de Trabalho\\gelo2.jpg").convert()
]

# Redimensionar backgrounds
backgrounds = [pygame.transform.scale(bg, (LARGURA, ALTURA)) for bg in backgrounds]

# Personagem
personagem = pygame.Rect(50, ALTURA - TAMANHO_PERSONAGEM[1], *TAMANHO_PERSONAGEM)
velocidade_y = 0
no_chao = True

# Variáveis de fundo
indice_fase = 0
inicio_fase = time.time()
scroll_x = 0

# Relógio
clock = pygame.time.Clock()

# Loop principal
while True:
    clock.tick(FPS)
    tempo_passado = time.time() - inicio_fase

    # Trocar fase a cada 60 segundos
    if tempo_passado > 60:
        indice_fase = (indice_fase + 1) % 3
        inicio_fase = time.time()
        scroll_x = 0

    # Eventos
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Teclas
    teclas = pygame.key.get_pressed()

    # Pulo
    if teclas[pygame.K_UP] and no_chao:
        velocidade_y = PULO
        no_chao = False

    # Agachar
    if teclas[pygame.K_DOWN] and no_chao:
        personagem.height = 50
    else:
        personagem.height = 100

    # Física do pulo
    personagem.y += velocidade_y
    velocidade_y += GRAVIDADE

    if personagem.y >= ALTURA - personagem.height:
        personagem.y = ALTURA - personagem.height
        velocidade_y = 0
        no_chao = True

    # Scroll do background
    scroll_x -= 2
    if scroll_x <= -LARGURA:
        scroll_x = 0

    # Desenhar fundo (loop infinito)
    tela.blit(backgrounds[indice_fase], (scroll_x, 0))
    tela.blit(backgrounds[indice_fase], (scroll_x + LARGURA, 0))

    # Desenhar personagem
    pygame.draw.rect(tela, (0, 0, 255), personagem)

    # Atualizar display
    pygame.display.flip()

    # Limpar tela
    tela.fill(BRANCO)

'''






'''
#jogo sem o "concentre-se" quando não estiver se movendo


import pygame
import sys
import time
import threading
from pylsl import StreamInlet, resolve_byprop
import numpy as np
from scipy.signal import butter, lfilter, welch

# EEG Configurações
fs = 512
window_size = 0.5
samples_per_window = int(fs * window_size)
EEG_comando = "neutro"

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

def band_power(signal):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

def ler_sinais_EEG():
    global EEG_comando
    streams = resolve_byprop('name', 'openvibeSignal')
    if not streams:
        print("Nenhum stream LSL encontrado.")
        return

    inlet = StreamInlet(streams[0])
    buffer = []

    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample)

        if len(buffer) >= samples_per_window:
            window = np.array(buffer[-samples_per_window:])
            window = np.array(window)
            c3 = window[:, 0]
            c4 = window[:, 1]

            mu_c3 = bandpass_filter(c3, 8, 13, fs)
            beta_c3 = bandpass_filter(c3, 13, 30, fs)
            mu_c4 = bandpass_filter(c4, 8, 13, fs)
            beta_c4 = bandpass_filter(c4, 13, 30, fs)

            mu_power_diff = band_power(mu_c4) - band_power(mu_c3)
            beta_power_diff = band_power(beta_c4) - band_power(beta_c3)

            if mu_power_diff > 0.5 or beta_power_diff > 0.5:
                EEG_comando = "abrir"
            elif mu_power_diff < -0.5 or beta_power_diff < -0.5:
                EEG_comando = "fechar"
            else:
                EEG_comando = "neutro"

# Iniciar thread de EEG
threading.Thread(target=ler_sinais_EEG, daemon=True).start()

# Inicializar o Pygame
pygame.init()

# Constantes
LARGURA, ALTURA = 800, 400
FPS = 60
TAMANHO_PERSONAGEM = (50, 100)
GRAVIDADE = 1
PULO = -20

# Cores
BRANCO = (255, 255, 255)

# Criar tela
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Jogo com 3 Fases e EEG")

# Carregar backgrounds
backgrounds = [
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\cidade.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\floresta.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\OneDrive\\Área de Trabalho\\gelo2.jpg").convert()
]

backgrounds = [pygame.transform.scale(bg, (LARGURA, ALTURA)) for bg in backgrounds]

# Personagem
personagem = pygame.Rect(50, ALTURA - TAMANHO_PERSONAGEM[1], *TAMANHO_PERSONAGEM)
velocidade_y = 0
no_chao = True

indice_fase = 0
inicio_fase = time.time()
scroll_x = 0

clock = pygame.time.Clock()

while True:
    clock.tick(FPS)
    tempo_passado = time.time() - inicio_fase
    if tempo_passado > 60:
        indice_fase = (indice_fase + 1) % 3
        inicio_fase = time.time()
        scroll_x = 0

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    teclas = pygame.key.get_pressed()

    # EEG substitui teclas
    if EEG_comando == "abrir" and no_chao:
        velocidade_y = PULO
        no_chao = False
    if EEG_comando == "fechar" and no_chao:
        personagem.height = 50
    else:
        personagem.height = 100

    personagem.y += velocidade_y
    velocidade_y += GRAVIDADE

    if personagem.y >= ALTURA - personagem.height:
        personagem.y = ALTURA - personagem.height
        velocidade_y = 0
        no_chao = True

    scroll_x -= 2
    if scroll_x <= -LARGURA:
        scroll_x = 0

    tela.blit(backgrounds[indice_fase], (scroll_x, 0))
    tela.blit(backgrounds[indice_fase], (scroll_x + LARGURA, 0))

    pygame.draw.rect(tela, (0, 0, 255), personagem)
    pygame.display.flip()
    tela.fill(BRANCO)

    
'''


import pygame
import sys
import time
import threading
from pylsl import StreamInlet, resolve_byprop
import numpy as np
from scipy.signal import butter, lfilter, welch

# EEG Configurações
fs = 512
window_size = 0.5
samples_per_window = int(fs * window_size)
EEG_comando = "neutro"
tempo_neutro = time.time()


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

def band_power(signal):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

def ler_sinais_EEG():
    global EEG_comando, tempo_neutro
    streams = resolve_byprop('name', 'openvibeSignal')
    if not streams:
        print("Nenhum stream LSL encontrado.")
        return

    inlet = StreamInlet(streams[0])
    buffer = []

    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample)

        if len(buffer) >= samples_per_window:
            window = np.array(buffer[-samples_per_window:])
            window = np.array(window)
            c3 = window[:, 0]
            c4 = window[:, 1]

            mu_c3 = bandpass_filter(c3, 8, 13, fs)
            beta_c3 = bandpass_filter(c3, 13, 30, fs)
            mu_c4 = bandpass_filter(c4, 8, 13, fs)
            beta_c4 = bandpass_filter(c4, 13, 30, fs)

            mu_power_diff = band_power(mu_c4) - band_power(mu_c3)
            beta_power_diff = band_power(beta_c4) - band_power(beta_c3)

            if mu_power_diff > 0.5 or beta_power_diff > 0.5:
                EEG_comando = "abrir"
            elif mu_power_diff < -0.5 or beta_power_diff < -0.5:
                EEG_comando = "fechar"
            else:
                EEG_comando = "neutro"

# Iniciar thread de EEG
threading.Thread(target=ler_sinais_EEG, daemon=True).start()

# Inicializar o Pygame
pygame.init()

# Constantes
LARGURA, ALTURA = 800, 400
FPS = 60
TAMANHO_PERSONAGEM = (50, 100)
GRAVIDADE = 1
PULO = -20

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

# Criar tela
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Jogo com 3 Fases e EEG")

# Fonte
fonte = pygame.font.SysFont(None, 48)

# Carregar backgrounds
backgrounds = [
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\cidade.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\\OneDrive\Área de Trabalho\\floresta.jpg").convert(),
    pygame.image.load("C:\\Users\\carlo\OneDrive\\Área de Trabalho\\gelo2.jpg").convert()
]

backgrounds = [pygame.transform.scale(bg, (LARGURA, ALTURA)) for bg in backgrounds]

# Personagem
personagem = pygame.Rect(50, ALTURA - TAMANHO_PERSONAGEM[1], *TAMANHO_PERSONAGEM)
velocidade_y = 0
no_chao = True

indice_fase = 0
inicio_fase = time.time()
scroll_x = 0
clock = pygame.time.Clock()
tempo_ultimo_neutro = time.time()

while True:
    clock.tick(FPS)
    tempo_passado = time.time() - inicio_fase
    if tempo_passado > 60:
        indice_fase = (indice_fase + 1) % 3
        inicio_fase = time.time()
        scroll_x = 0

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # EEG substitui teclas
    if EEG_comando == "abrir" and no_chao:
        velocidade_y = PULO
        no_chao = False
        tempo_ultimo_neutro = time.time()
    elif EEG_comando == "fechar" and no_chao:
        personagem.height = 50
        tempo_ultimo_neutro = time.time()
    else:
        personagem.height = 100
        if EEG_comando == "neutro":
            if time.time() - tempo_ultimo_neutro > 2:
                mensagem = fonte.render("Concentre-se", True, PRETO)
                tela.blit(mensagem, (LARGURA//2 - mensagem.get_width()//2, 50))
        else:
            tempo_ultimo_neutro = time.time()

    personagem.y += velocidade_y
    velocidade_y += GRAVIDADE

    if personagem.y >= ALTURA - personagem.height:
        personagem.y = ALTURA - personagem.height
        velocidade_y = 0
        no_chao = True

    scroll_x -= 2
    if scroll_x <= -LARGURA:
        scroll_x = 0

    tela.blit(backgrounds[indice_fase], (scroll_x, 0))
    tela.blit(backgrounds[indice_fase], (scroll_x + LARGURA, 0))

    pygame.draw.rect(tela, (0, 0, 255), personagem)

    if EEG_comando == "neutro" and time.time() - tempo_ultimo_neutro > 2:
        mensagem = fonte.render("Concentre-se", True, PRETO)
        tela.blit(mensagem, (LARGURA//2 - mensagem.get_width()//2, 50))

    pygame.display.flip()
    tela.fill(BRANCO)
