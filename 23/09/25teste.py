import numpy as np
import pandas as pd
import time
import pygame  # <-- Importação adicionada
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pylsl import resolve_byprop, StreamInlet

# --- Função para criar arquivos CSV de exemplo (inalterada) ---
def criar_csv_exemplo(n_canais=14, fs=250, duracao_seg=10):
    """Gera arquivos CSV de exemplo para o treinamento."""
    n_amostras = fs * duracao_seg
    colunas_canais = [f'canal_{i+1}' for i in range(n_canais)]
    dados_esquerda = np.random.randn(n_amostras, n_canais) * 10 + 0.5
    df_esquerda = pd.DataFrame(dados_esquerda, columns=colunas_canais)
    df_esquerda.to_csv('treino_esquerda.csv', index=False)
    print("📄 Arquivo 'treino_esquerda.csv' criado.")
    dados_direita = np.random.randn(n_amostras, n_canais) * 10 - 0.5
    df_direita = pd.DataFrame(dados_direita, columns=colunas_canais)
    df_direita.to_csv('treino_direita.csv', index=False)
    print("📄 Arquivo 'treino_direita.csv' criado.")


# --- Classes EEGStream, EEGDataLoaderCSV, EEGPreprocessador, EEGClassificador (inalteradas) ---
class EEGStream:
    """Gerencia a conexão e coleta de dados de um stream EEG LSL."""
    def __init__(self, config: dict):
        self.config = config
        print(f"🔍 Procurando stream LSL com nome '{self.config['LSL_STREAM_NAME']}'...")
        streams = resolve_byprop('name', self.config['LSL_STREAM_NAME'], 1, 5)
        if not streams:
            raise RuntimeError(f"❌ Nenhum stream LSL com nome '{self.config['LSL_STREAM_NAME']}' encontrado! Verifique o OpenViBE.")
        
        self.inlet = StreamInlet(streams[0])
        
        info = self.inlet.info()
        channel_count = info.channel_count()
        
        if self.config["N_CANAIS"] != channel_count:
            print(f"⚠️  Aviso: O número de canais foi ajustado de {self.config['N_CANAIS']} para {channel_count} (detectado do stream).")
            self.config["N_CANAIS"] = channel_count
            
        print(f"✅ Stream EEG '{info.name()}' encontrado com {self.config['N_CANAIS']} canais!")

    def get_chunk(self) -> np.ndarray | None:
        """Puxa um chunk de dados do stream LSL."""
        samples, timestamps = self.inlet.pull_chunk()
        if not samples:
            return None
        return np.array(samples).T

    def close(self):
        """Fecha a conexão do inlet."""
        if self.inlet:
            self.inlet.close_stream()
            print("🔌 Conexão LSL fechada.")


class EEGDataLoaderCSV:
    """Carrega dados de EEG a partir de arquivos CSV para o treinamento."""
    def __init__(self, config: dict):
        self.n_canais = config["N_CANAIS"]

    def carregar_dados(self, caminho_arquivo: str) -> np.ndarray:
        try:
            df = pd.read_csv(caminho_arquivo)
            dados = df.iloc[:, :self.n_canais].values.T
            print(f"✅ Dados de treino carregados de '{caminho_arquivo}'. Shape: {dados.shape}")
            return dados
        except FileNotFoundError:
            raise RuntimeError(f"❌ Arquivo não encontrado: {caminho_arquivo}")
        except IndexError:
             raise IndexError(f"❌ O arquivo '{caminho_arquivo}' não possui {self.n_canais} colunas. Verifique o arquivo ou o parâmetro N_CANAIS.")


class EEGPreprocessador:
    """Cria épocas a partir dos sinais."""
    def __init__(self, config: dict):
        self.janela = config["JANELA"]

    def criar_epocas_treino(self, sinal: np.ndarray, classe: int):
        X, y = [], []
        n_amostras = sinal.shape[1]
        for i in range(0, n_amostras - self.janela, self.janela):
            X.append(sinal[:, i:i + self.janela])
            y.append(classe)
        return X, y


class EEGClassificador:
    """Treina e classifica sinais EEG."""
    def __init__(self, config: dict):
        modelo = config["MODELO"].upper()
        self.csp = CSP(n_components=config["N_COMPONENTES_CSP"], reg=None, log=True, norm_trace=False)
        
        if modelo == "SVM":
            self.clf = Pipeline([("CSP", self.csp), ("SVM", SVC(kernel="rbf", C=1))])
        else: # LDA como padrão
            self.clf = Pipeline([("CSP", self.csp), ("LDA", LDA())])
        print(f"Classificador inicializado com modelo: {modelo}")

    def treinar(self, X: np.ndarray, y: np.ndarray):
        print("\n🚀 Treinando classificador...")
        self.clf.fit(X, y)
        print("✅ Treinamento concluído!")

    def prever(self, epoca: np.ndarray) -> int:
        epoca_formatada = epoca[np.newaxis, :, :]
        return self.clf.predict(epoca_formatada)[0]


# --- Pipeline principal, agora com o método do Jogo ---
class EEGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.eeg_stream = EEGStream(self.config)
        self.loader = EEGDataLoaderCSV(self.config)
        self.preprocessador = EEGPreprocessador(self.config)
        self.classificador = EEGClassificador(self.config)

    def treinar(self):
        print("\n=== Fase de Treinamento ===")
        sinal_esquerda = self.loader.carregar_dados(self.config["ARQUIVO_TREINO_ESQUERDA"])
        sinal_direita = self.loader.carregar_dados(self.config["ARQUIVO_TREINO_DIREITA"])
        X_e, y_e = self.preprocessador.criar_epocas_treino(sinal_esquerda, 0) # 0 para Esquerda
        X_d, y_d = self.preprocessador.criar_epocas_treino(sinal_direita, 1)   # 1 para Direita
        X = np.array(X_e + X_d)
        y = np.array(y_e + y_d)
        self.classificador.treinar(X, y)

    def iniciar_jogo_pygame(self):
        """
        Inicia um loop de jogo com Pygame controlado pelas predições do EEG.
        """
        print("\n=== 🎮 Iniciando Jogo com Controle EEG ===")

        # --- Configuração do Pygame ---
        pygame.init()
        LARGURA, ALTURA = 800, 600
        screen = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption("Controle EEG com Pygame")
        clock = pygame.time.Clock()
        
        # --- Configuração do Jogador ---
        player = pygame.Rect(LARGURA / 2 - 25, ALTURA - 100, 50, 50)
        VELOCIDADE_JOGADOR = 10

        try:
            buffer_dados = np.zeros((self.config["N_CANAIS"], 0))
            running = True
            
            while running:
                # --- Tratamento de Eventos (ex: fechar janela) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # --- Lógica de Coleta e Classificação do EEG ---
                chunk_novo = self.eeg_stream.get_chunk()
                if chunk_novo is not None and chunk_novo.shape[1] > 0:
                    buffer_dados = np.hstack([buffer_dados, chunk_novo])

                # Processa o buffer se houver dados para uma época
                while buffer_dados.shape[1] >= self.config["JANELA"]:
                    epoca_atual = buffer_dados[:, -self.config["JANELA"]:]
                    predicao = self.classificador.prever(epoca_atual)
                    
                    # --- ATUALIZA A POSIÇÃO DO JOGADOR COM BASE NA PREDIÇÃO ---
                    if predicao == 0: # Esquerda
                        player.x -= VELOCIDADE_JOGADOR
                        print(f"\rPredição: ⬅️  Esquerda", end="")
                    elif predicao == 1: # Direita
                        player.x += VELOCIDADE_JOGADOR
                        print(f"\rPredição: ➡️  Direita ", end="")
                    
                    # Desliza a janela do buffer
                    buffer_dados = buffer_dados[:, self.config["PASSO_JANELA"]:]

                # --- Lógica do Jogo (limites de tela) ---
                if player.left < 0:
                    player.left = 0
                if player.right > LARGURA:
                    player.right = LARGURA

                # --- Renderização ---
                screen.fill((25, 25, 112))  # Fundo azul escuro
                pygame.draw.rect(screen, (0, 255, 0), player) # Jogador verde
                pygame.display.flip() # Atualiza a tela

                clock.tick(60) # Limita o jogo a 60 FPS

        except Exception as e:
            print(f"\n❌ Ocorreu um erro no jogo: {e}")
        finally:
            pygame.quit()
            self.eeg_stream.close()
            print("\nJogo encerrado.")


if __name__ == "__main__":
    # --- CONFIGURAÇÃO CENTRAL ---
    CONFIG = {
        "ARQUIVO_TREINO_ESQUERDA": "treino_esquerda.csv",
        "ARQUIVO_TREINO_DIREITA": "treino_direita.csv",
        "LSL_STREAM_NAME": 'openvibeSignal',
        "N_CANAIS": 14,
        "JANELA": 250,
        "PASSO_JANELA": 50,
        "MODELO": "LDA",
        "N_COMPONENTES_CSP": 6
    }

    # Gerar arquivos CSV de exemplo, se necessário.
    # criar_csv_exemplo(n_canais=CONFIG["N_CANAIS"])

    try:
        pipeline = EEGPipeline(config=CONFIG)
        pipeline.treinar()
        
        input("\n▶️ Pressione Enter para iniciar o jogo...")
        pipeline.iniciar_jogo_pygame() # <-- Chamada para o novo método

    except Exception as e:
        print(f"\n\n🚨 Um erro crítico ocorreu: {e}")