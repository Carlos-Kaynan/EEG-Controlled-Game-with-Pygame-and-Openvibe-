import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, welch
from scipy.integrate import trapezoid
import time
import threading

# --- NOVA IMPORTAÇÃO PARA LSL ---
from pylsl import StreamInlet, resolve_byprop

import pygame
import random
import sys
import os

# ===================================================================
# PARTE 1: LÓGICA DO CLASSIFICADOR DE EEG (MODIFICADA PARA LSL)
# ===================================================================

# Variável global para comunicar o comando do EEG para o jogo
comando_eeg = "PARADO"

# --- CONFIGURAÇÃO DOS CANAIS E JANELA ---
# ATENÇÃO: Verifique os índices (base 0) dos seus canais no OpenVibe!
# Ex: Se seus canais são os 10º, 14º e 15º na lista, os índices são 9, 13, 14.
CANAIS_INDICES = [9, 13, 14]
JANELA_AMOSTRAS = 250  # 1 segundo de dados a 250 Hz
PASSO_JANELA = 125     # 50% de sobreposição (0.5 segundos)

# ========= Funções Auxiliares de EEG (sem alterações) =========

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
    df.columns = df.columns.map(str)
    canais_desejados = [c for c in ['10', '14', '15'] if c in df.columns]
    if not canais_desejados:
        raise ValueError(f"Canais C3, CZ e C4 não encontrados! Colunas disponíveis: {list(df.columns)}")
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

# --- FUNÇÃO DE CLASSIFICAÇÃO TOTALMENTE REESCRITA PARA LSL ---
def classificar_eeg_em_thread(clf, scaler):
    """
    Função que será executada na thread secundária.
    Conecta-se ao LSL, coleta dados em tempo real e atualiza 'comando_eeg'.
    """
    global comando_eeg

    try:
        print("Procurando stream EEG do OpenVibe ('openvibeSignal')...")
        streams = resolve_byprop('name', 'openvibeSignal', timeout=5)
        if not streams:
            raise RuntimeError("Erro: Nenhum stream 'openvibeSignal' encontrado. Verifique o OpenVibe.")
        
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        print(f"✅ Stream encontrado: {info.name()} @ {info.nominal_srate()} Hz com {info.channel_count()} canais.")
    except Exception as e:
        print(e)
        comando_eeg = "ERRO_STREAM"
        return

    data_buffer = []
    
    print("\n⏱️  Thread de classificação EEG via LSL iniciada...\n")
    while True:
        # 1. Puxa um chunk de novas amostras do LSL
        samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=100)
        
        if samples:
            # 2. Adiciona as novas amostras ao buffer
            data_buffer.extend(samples)

            # 3. Se o buffer tiver dados suficientes para uma janela
            if len(data_buffer) >= JANELA_AMOSTRAS:
                # 4. Pega a janela de dados mais recente (as últimas N amostras)
                janela_dados = np.array(data_buffer[-JANELA_AMOSTRAS:])
                
                # 5. Seleciona apenas os canais de interesse
                dados_canais = janela_dados[:, CANAIS_INDICES]
                
                # 6. Cria um DataFrame para compatibilidade com a função de features
                df_janela = pd.DataFrame(dados_canais, columns=['10', '14', '15'])

                # 7. Extrai as features e faz a predição
                feat_janela, _ = extrair_features_janelas(df_janela, janela=JANELA_AMOSTRAS, scaler=scaler)
                
                if feat_janela.size > 0:
                    pred = clf.predict(feat_janela)[0]
                    if pred == 0:
                        comando_eeg = "ESQUERDA"
                        print("🧠 COMANDO: ESQUERDA")
                    else:
                        comando_eeg = "DIREITA"
                        print("🧠 COMANDO: DIREITA")

                # 8. Remove dados antigos do buffer para criar o efeito de "janela deslizante"
                # Remove o número de amostras correspondente ao passo da janela
                del data_buffer[:PASSO_JANELA]
        
        # Pequena pausa para não sobrecarregar a CPU
        time.sleep(0.1)


# ========= PRÉ-PROCESSAMENTO E TREINAMENTO (Executado uma vez no início) =========
print("--- Iniciando Treinamento do Classificador com arquivos CSV ---")
# Use os seus caminhos absolutos ou coloque os arquivos na mesma pasta do script
arquivo_esquerda = "C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\recordAlphaBeta-[2025.10.14-17.27.17]Esquerda.csv"
arquivo_direita = "C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\recordAlphaBeta-[2025.10.14-17.24.46]Direita.csv"
# O arquivo_teste não é mais necessário para a classificação em tempo real
# arquivo_teste = "C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\recordAlphaBeta-[2025.10.14-17.30.18]DireitaEsquerda.csv"

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

# ===================================================================
# PARTE 2: LÓGICA DO JOGO PYGAME (Com ajustes solicitados)
# ===================================================================
pygame.init()

LARGURA_TELA = 600
ALTURA_TELA = 800
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Controle BCI - Desvie dos Carros")

BRANCO = (255, 255, 255); PRETO = (0, 0, 0); CINZA = (100, 100, 100)
AMARELO = (255, 255, 0); VERMELHO = (200, 0, 0)

carro_jogador_largura, carro_jogador_altura = 60, 90
carro_inimigo_largura, carro_inimigo_altura = 60, 90

try:
    caminho_jogador = "C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\Car_game\\player.png"
    caminho_inimigo = "C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\Car_game\\enemi.png"
    imagem_carro_jogador = pygame.transform.scale(pygame.image.load(caminho_jogador), (carro_jogador_largura, carro_jogador_altura))
    imagem_carro_inimigo = pygame.transform.scale(pygame.image.load(caminho_inimigo), (carro_inimigo_largura, carro_inimigo_altura))
except pygame.error as e:
    print(f"Erro ao carregar imagem: {e}"); pygame.quit(); sys.exit()

pos_x_jogador = (LARGURA_TELA - carro_jogador_largura) // 2
pos_y_jogador = ALTURA_TELA - carro_jogador_altura - 20
velocidade_jogador = 8
velocidade_inimigo = 3 # Velocidade reduzida como solicitado

fonte = pygame.font.SysFont(None, 50)
fonte_hud = pygame.font.SysFont(None, 36)
relogio = pygame.time.Clock()

# --- Novas variáveis para pista animada ---
offset_linha = 0
velocidade_pista = 6  # ajuste para aumentar/diminuir percepção de movimento

def desenhar_pista():
    """Desenha a pista com faixas amarelas em movimento (efeito de estrada rolando)."""
    global offset_linha
    tela.fill(CINZA)
    espacamento = 40
    for i in range(ALTURA_TELA // espacamento + 3):
        y = (i * espacamento + offset_linha) % (ALTURA_TELA + espacamento) - espacamento
        pygame.draw.rect(tela, AMARELO, (LARGURA_TELA / 2 - 5, y, 10, 20))
    offset_linha += velocidade_pista
    if offset_linha >= espacamento:
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

def loop_jogo():
    global pos_x_jogador
    pontuacao = 0; colisoes = 0; DURACAO_JOGO = 60  # 1 minuto
    tempo_inicial = pygame.time.get_ticks()
    jogo_ativo = True
    contador_spawn_inimigo = 0
    carros_inimigos = []

    while jogo_ativo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        
        if comando_eeg == "ERRO_STREAM":
            # Lida com o erro de não encontrar o stream
            tela.fill(PRETO)
            texto_erro = fonte_hud.render("Erro: Stream LSL não encontrado. Verifique o OpenVibe.", True, VERMELHO)
            texto_rect = texto_erro.get_rect(center=(LARGURA_TELA/2, ALTURA_TELA/2))
            tela.blit(texto_erro, texto_rect)
            pygame.display.flip()
            pygame.time.wait(5000)
            jogo_ativo = False
            continue

        tempo_decorrido = (pygame.time.get_ticks() - tempo_inicial) / 1000
        if tempo_decorrido >= DURACAO_JOGO:
            jogo_ativo = False
        
        # Controle EXCLUSIVO pelo EEG (sem teclado)
        if comando_eeg == "ESQUERDA":
            pos_x_jogador -= velocidade_jogador
        elif comando_eeg == "DIREITA":
            pos_x_jogador += velocidade_jogador
        # comando_eeg == "PARADO" -> sem movimento

        # Mantém jogador dentro da tela
        pos_x_jogador = max(0, min(pos_x_jogador, LARGURA_TELA - carro_jogador_largura))

        # Spawn de inimigos
        if contador_spawn_inimigo % 100 == 0:
            pos_x_inimigo = random.randint(0, LARGURA_TELA - carro_inimigo_largura)
            carros_inimigos.append({'x': pos_x_inimigo, 'y': -carro_inimigo_altura})
        contador_spawn_inimigo += 1
        
        # CAIXAS DE COLISÃO REDUZIDAS (menos sensíveis)
        # Jogador: inset na esquerda/direita e em cima/baixo
        jogador_hitbox = pygame.Rect(
            pos_x_jogador + int(carro_jogador_largura * 0.15),
            pos_y_jogador + int(carro_jogador_altura * 0.12),
            int(carro_jogador_largura * 0.7),
            int(carro_jogador_altura * 0.75)
        )

        for inimigo in carros_inimigos[:]:
            inimigo['y'] += velocidade_inimigo
            inimigo_hitbox = pygame.Rect(
                inimigo['x'] + int(carro_inimigo_largura * 0.12),
                inimigo['y'] + int(carro_inimigo_altura * 0.12),
                int(carro_inimigo_largura * 0.76),
                int(carro_inimigo_altura * 0.76)
            )

            if jogador_hitbox.colliderect(inimigo_hitbox):
                colisoes += 1
                # remove o inimigo que colidiu (comportamento solicitado)
                try:
                    carros_inimigos.remove(inimigo)
                except ValueError:
                    pass
            elif inimigo['y'] > ALTURA_TELA:
                pontuacao += 1
                try:
                    carros_inimigos.remove(inimigo)
                except ValueError:
                    pass
        
        # Desenha tudo
        desenhar_elementos(pos_x_jogador, pos_y_jogador, carros_inimigos)
        
        # HUD tempo/pontos/colisões
        tempo_restante = max(0, DURACAO_JOGO - tempo_decorrido)
        texto_tempo = fonte_hud.render(f"Tempo: {int(tempo_restante)}", True, BRANCO)
        texto_pontos = fonte_hud.render(f"Pontos: {pontuacao}", True, AMARELO)
        texto_colisoes_hud = fonte_hud.render(f"Colisões: {colisoes}", True, VERMELHO)
        tela.blit(texto_tempo, (10, 10))
        tela.blit(texto_pontos, (LARGURA_TELA - texto_pontos.get_width() - 10, 10))
        tela.blit(texto_colisoes_hud, (LARGURA_TELA - texto_colisoes_hud.get_width() - 10, 45))
        
        pygame.display.flip()
        relogio.tick(60)

    mostrar_mensagem_final(pontuacao, colisoes)
    pygame.quit()
    sys.exit()

# ===================================================================
# PARTE 3: EXECUÇÃO PRINCIPAL (MODIFICADA)
# ===================================================================
if __name__ == '__main__':
    # 1. Cria a thread para o classificador de EEG
    eeg_thread = threading.Thread(
        target=classificar_eeg_em_thread, 
        args=(clf, scaler), 
        daemon=True
    )

    # 2. Inicia a thread do EEG.
    eeg_thread.start()

    # 3. Inicia o loop do jogo na thread principal.
    loop_jogo()
