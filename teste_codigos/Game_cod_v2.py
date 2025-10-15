import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, welch
from scipy.integrate import trapezoid
import time
import threading

# LSL
from pylsl import StreamInlet, resolve_byprop

# Pygame / jogo
import pygame
import random
import sys
import os

# ---------------------------
# PARTE 1: Parâmetros / Globals
# ---------------------------
comando_eeg = "PARADO"  # valor atualizado pela thread do EEG ("ESQUERDA"/"DIREITA"/"ERRO_STREAM"/"PARADO")

# Canais e janelas (mantenha como no seu código)
CANAIS_INDICES = [1, 2] 
JANELA_AMOSTRAS = 250  # 1 segundo a 250 Hz
PASSO_JANELA = 125     # 50% overlap

# ---------------------------
# PARTE 2: Funções EEG / features
# ---------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=8, highcut=28, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data, axis=0)

def selecionar_canais(df):
    # tenta pegar as colunas nomeadas '14' e '15' como no seu pipeline original
    df.columns = df.columns.map(str)
    canais_desejados = [c for c in ['14', '15'] if c in df.columns]
    if not canais_desejados:
        raise ValueError(f"Canais '14' e '15' não encontrados! Colunas disponíveis: {list(df.columns)}")
    return df[canais_desejados]

def calcular_bandpower(x, fs=250, fmin=8, fmax=28):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256))
    mask = (f >= fmin) & (f <= fmax)
    return trapezoid(Pxx[mask], f[mask])

def extrair_features_janelas(df, janela=JANELA_AMOSTRAS, sobreposicao=0.5, scaler=None):
    df_selecionado = selecionar_canais(df)
    dados = df_selecionado.values
    imputer = SimpleImputer(strategy='mean')
    dados = imputer.fit_transform(dados)
    if scaler is None:
        scaler = StandardScaler()
        dados = scaler.fit_transform(dados)
    else:
        dados = scaler.transform(dados)
    dados_filtrados = bandpass_filter(dados)
    passo = int(janela * (1 - sobreposicao))
    janelas = [dados_filtrados[i:i+janela] for i in range(0, max(1, len(dados_filtrados) - janela + 1), passo)]
    features = []
    for j in janelas:
        if j.shape[0] < janela: continue
        feat_j = []
        for c in range(j.shape[1]):
            canal = j[:, c]
            feat_j.append(np.mean(canal**2))
            feat_j.append(np.std(canal))
            feat_j.append(calcular_bandpower(canal))
        features.append(feat_j)
    return np.array(features), scaler

# ---------------------------
# PARTE 3: Thread de classificação em tempo real via LSL
# ---------------------------
def classificar_eeg_em_thread(clf, scaler):
    """
    Thread que resolve o stream 'OV_EEG', lê chunks, extrai features e atualiza `comando_eeg`.
    """
    global comando_eeg

    try:
        print("Procurando stream EEG do OpenVibe ('OV_EEG')...")
        streams = resolve_byprop('name', 'OV_EEG', timeout=5)
        if not streams:
            raise RuntimeError("Nenhum stream 'OV_EEG' encontrado.")
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        print(f"✅ Stream encontrado: {info.name()} @ {info.nominal_srate()} Hz com {info.channel_count()} canais.")
    except Exception as e:
        print("Erro ao conectar LSL:", e)
        comando_eeg = "ERRO_STREAM"
        return

    data_buffer = []
    print("Thread EEG iniciada (LSL -> classificação)...")

    while True:
        samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=100)
        if samples:
            data_buffer.extend(samples)
            # garantia: usar somente as últimas JANELA_AMOSTRAS amostras para cada janela
            if len(data_buffer) >= JANELA_AMOSTRAS:
                janela_dados = np.array(data_buffer[-JANELA_AMOSTRAS:])
                # seleciona canais definidos pelo índice
                try:
                    dados_canais = janela_dados[:, CANAIS_INDICES]
                except Exception as e:
                    print("Erro ao indexar canais:", e)
                    comando_eeg = "ERRO_STREAM"
                    return

                # Cria DataFrame com nomes compatíveis
                # Obs: a função extrair_features_janelas espera colunas '14' e '15', então damos esses nomes
                df_janela = pd.DataFrame(dados_canais, columns=['14', '15'])

                feat_janela, _ = extrair_features_janelas(df_janela, janela=JANELA_AMOSTRAS, scaler=scaler)
                if feat_janela.size > 0:
                    pred = clf.predict(feat_janela)[0]
                    if pred == 0:
                        comando_eeg = "ESQUERDA"
                        # print("🧠 COMANDO: ESQUERDA")
                    else:
                        comando_eeg = "DIREITA"
                        # print("🧠 COMANDO: DIREITA")

                # avança no buffer (passo deslizante)
                del data_buffer[:PASSO_JANELA]

        time.sleep(0.05)  # pequeno sono para aliviar CPU

# ---------------------------
# PARTE 4: Treinamento (executa no start)
# ---------------------------
print("--- Iniciando Treinamento do Classificador com arquivos CSV ---")
arquivo_esquerda = "C:\\Users\\igo_p\\Desktop\\game_cod\\recordAlphaBeta-[2025.10.14-17.27.17]Esquerda.csv"
arquivo_direita = "C:\\Users\\igo_p\\Desktop\\game_cod\\recordAlphaBeta-[2025.10.14-17.24.46]Direita.csv"

df_esq = pd.read_csv(arquivo_esquerda)
df_dir = pd.read_csv(arquivo_direita)

feat_esq, scaler = extrair_features_janelas(df_esq)
feat_dir, _ = extrair_features_janelas(df_dir, scaler=scaler)

min_len = min(len(feat_esq), len(feat_dir))
X_train = np.vstack([feat_esq[:min_len], feat_dir[:min_len]])
y_train = np.hstack([np.zeros(min_len), np.ones(min_len)])

clf = LDA()
clf.fit(X_train, y_train)
print("✅ Classificador LDA treinado com sucesso!")
print("---------------------------------------------")



# ---------------------------
# PARTE 5: Jogo Pygame (controlado apenas por EEG)
# ---------------------------

pygame.init()

LARGURA_TELA = 500
ALTURA_TELA = 700
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.setCaption("Controle BCI - Desvie dos Carros")

BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (100, 100, 100)
AMARELO = (255, 255, 0)
VERMELHO = (200, 0, 0)

carro_jogador_largura, carro_jogador_altura = 60, 90
carro_inimigo_largura, carro_inimigo_altura = 60, 90

try:
    caminho_jogador = "C:\\Users\\igo_p\\Desktop\\game_cod\\player.png"
    caminho_inimigo = "C:\\Users\\igo_p\\Desktop\\game_cod\\enemi.png"
    imagem_carro_jogador = pygame.transform.scale(pygame.image.load(caminho_jogador), (carro_jogador_largura, carro_jogador_altura))
    imagem_carro_inimigo = pygame.transform.scale(pygame.image.load(caminho_inimigo), (carro_inimigo_largura, carro_inimigo_altura))
except pygame.error as e:
    print(f"Erro ao carregar imagem: {e}")
    pygame.quit()
    sys.exit()

pos_x_jogador = (LARGURA_TELA - carro_jogador_largura) // 2
pos_y_jogador = ALTURA_TELA - carro_jogador_altura - 20
velocidade_jogador = 8
velocidade_inimigo = 3

fonte = pygame.font.SysFont(None, 50)
fonte_hud = pygame.font.SysFont(None, 36)
relogio = pygame.time.Clock()

# Variável para animar a pista
offset_linha = 0
velocidade_pista = 5  # quanto maior, mais rápido o chão "rola"

def desenhar_pista():
    global offset_linha
    tela.fill(CINZA)

    # desenha faixas amarelas descendo
    for i in range(ALTURA_TELA // 40 + 2):
        y = (i * 40 + offset_linha) % ALTURA_TELA
        pygame.draw.rect(tela, AMARELO, (LARGURA_TELA / 2 - 5, y, 10, 20))

    # atualiza o deslocamento (movimento)
    offset_linha += velocidade_pista
    if offset_linha >= 40:
        offset_linha = 0

def desenhar_elementos(x_jogador, y_jogador, inimigos):
    desenhar_pista()
    tela.blit(imagem_carro_jogador, (x_jogador, y_jogador))
    for inimigo in inimigos:
        tela.blit(imagem_carro_inimigo, (inimigo['x'], inimigo['y']))

def mostrar_mensagem_final(pontuacao, colisoes):
    tela.fill(PRETO)
    texto_fim = fonte.render("TEMPO ESGOTADO", True, BRANCO)
    texto_pontuacao = fonte.render(f"Pontuação Final: {pontuacao}", True, AMARELO)
    texto_colisoes = fonte.render(f"Colisões: {colisoes}", True, VERMELHO)
    tela.blit(texto_fim, texto_fim.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2 - 80)))
    tela.blit(texto_pontuacao, texto_pontuacao.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2)))
    tela.blit(texto_colisoes, texto_colisoes.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2 + 80)))
    pygame.display.flip()
    pygame.time.wait(5000)
