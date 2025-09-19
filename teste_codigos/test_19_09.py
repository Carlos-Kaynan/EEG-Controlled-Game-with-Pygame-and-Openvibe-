import numpy as np
import pandas as pd
import time
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# --- Função para criar arquivos CSV de exemplo ---
def criar_csv_exemplo(n_canais=14, fs=250, duracao_seg=10):
    """
    Gera arquivos CSV de exemplo:
    1. Treino para mão esquerda.
    2. Treino para mão direita.
    3. Um arquivo com dados mistos para classificação.
    """
    n_amostras = fs * duracao_seg
    colunas_canais = [f'canal_{i+1}' for i in range(n_canais)]

    # 1. Arquivo de treino - Mão Esquerda
    dados_esquerda = np.random.randn(n_amostras, n_canais) * 10 + 0.5
    df_esquerda = pd.DataFrame(dados_esquerda, columns=colunas_canais)
    df_esquerda.to_csv('treino_esquerda.csv', index=False)
    print("📄 Arquivo 'treino_esquerda.csv' criado.")

    # 2. Arquivo de treino - Mão Direita
    dados_direita = np.random.randn(n_amostras, n_canais) * 10 - 0.5
    df_direita = pd.DataFrame(dados_direita, columns=colunas_canais)
    df_direita.to_csv('treino_direita.csv', index=False)
    print("📄 Arquivo 'treino_direita.csv' criado.")
    
    # 3. Arquivo de classificação com dados mistos
    n_amostras_misto = fs * (duracao_seg // 2)
    dados_mistos_esq = np.random.randn(n_amostras_misto, n_canais) * 10 + 0.5
    dados_mistos_dir = np.random.randn(n_amostras_misto, n_canais) * 10 - 0.5
    dados_mistos_total = np.vstack([dados_mistos_esq, dados_mistos_dir]) # Empilha os dados
    df_misto = pd.DataFrame(dados_mistos_total, columns=colunas_canais)
    df_misto.to_csv('dados_classificacao_mista.csv', index=False)
    print("📄 Arquivo 'dados_classificacao_mista.csv' criado.")


class EEGDataLoaderCSV:
    """Classe para carregar dados de EEG a partir de arquivos CSV."""
    def __init__(self, n_canais: int = 14):
        self.n_canais = n_canais

    def carregar_dados(self, caminho_arquivo: str) -> np.ndarray:
        """Carrega dados de um único arquivo CSV."""
        try:
            df = pd.read_csv(caminho_arquivo)
            dados = df.iloc[:, :self.n_canais].values.T
            print(f"✅ Dados carregados de '{caminho_arquivo}'. Shape: {dados.shape}")
            return dados
        except FileNotFoundError:
            raise RuntimeError(f"❌ Arquivo não encontrado: {caminho_arquivo}")


class EEGPreprocessador:
    """Classe para criar épocas a partir dos sinais brutos."""
    def __init__(self, janela: int = 250):
        self.janela = janela

    def criar_epocas_treino(self, sinal: np.ndarray, classe: int):
        """Cria épocas e rótulos para o treinamento."""
        X, y = [], []
        n_amostras = sinal.shape[1]
        for i in range(0, n_amostras - self.janela, self.janela):
            X.append(sinal[:, i:i + self.janela])
            y.append(classe)
        return X, y

    def criar_epocas_classificacao(self, sinal: np.ndarray):
        """Cria épocas a partir de um sinal para classificação (sem rótulos)."""
        X = []
        n_amostras = sinal.shape[1]
        for i in range(0, n_amostras - self.janela, self.janela):
            X.append(sinal[:, i:i + self.janela])
        return np.array(X)


class EEGClassificador:
    """Classe para treinar e classificar sinais EEG."""
    def __init__(self, n_componentes: int = 6, modelo: str = "LDA"):
        self.csp = CSP(n_components=n_componentes, reg=None, log=True, norm_trace=False)
        if modelo.upper() == "SVM":
            self.clf = Pipeline([("CSP", self.csp), ("SVM", SVC(kernel="rbf", C=1))])
        else:
            self.clf = Pipeline([("CSP", self.csp), ("LDA", LDA())])

    def treinar(self, X: np.ndarray, y: np.ndarray):
        print("\n🚀 Treinando classificador...")
        self.clf.fit(X, y)
        print("✅ Treinamento concluído!")

    def prever(self, epoca: np.ndarray) -> int:
        epoca_formatada = epoca[np.newaxis, :, :]
        return self.clf.predict(epoca_formatada)[0]


class EEGPipeline:
    """Pipeline completo para treino e classificação a partir de arquivos CSV."""
    def __init__(self, caminho_esquerda: str, caminho_direita: str, caminho_classificacao: str,
                 fs=250, n_canais=14, janela=250, modelo="LDA"):
        self.caminho_esquerda = caminho_esquerda
        self.caminho_direita = caminho_direita
        self.caminho_classificacao = caminho_classificacao # Novo caminho
        
        self.loader = EEGDataLoaderCSV(n_canais)
        self.preprocessador = EEGPreprocessador(janela)
        self.classificador = EEGClassificador(modelo=modelo)

    def treinar(self):
        """Fase de treinamento."""
        print("\n=== Fase de Treinamento ===")
        sinal_esquerda = self.loader.carregar_dados(self.caminho_esquerda)
        sinal_direita = self.loader.carregar_dados(self.caminho_direita)

        X_e, y_e = self.preprocessador.criar_epocas_treino(sinal_esquerda, 0) # 0 para Esquerda
        X_d, y_d = self.preprocessador.criar_epocas_treino(sinal_direita, 1)  # 1 para Direita

        X = np.array(X_e + X_d)
        y = np.array(y_e + y_d)
        self.classificador.treinar(X, y)

    def iniciar_classificacao(self):
        """Inicia a classificação no arquivo de teste/aleatório."""
        print(f"\n=== Classificando o arquivo: {self.caminho_classificacao} ===")
        sinal = self.loader.carregar_dados(self.caminho_classificacao)
        epocas = self.preprocessador.criar_epocas_classificacao(sinal)

        if epocas.shape[0] == 0:
            print("⚠️ Nenhum dado para classificar. Verifique o arquivo.")
            return

        for i, epoca in enumerate(epocas):
            predicao = self.classificador.prever(epoca)
            resultado = "🖐️ Esquerda" if predicao == 0 else "🖐️ Direita"
            print(f"Época {i+1}: Previsto -> {resultado}")
            time.sleep(0.1)


if __name__ == "__main__":
    # 1. Gerar arquivos CSV de exemplo. Comente esta linha ao usar seus arquivos.
    criar_csv_exemplo()

    # 2. Definir os caminhos para os arquivos de dados.
    #    ↓↓↓ EDITE OS TRÊS CAMINHOS ABAIXO COM SEUS ARQUIVOS ↓↓↓
    ARQUIVO_TREINO_ESQUERDA = "C:\\Users\\User\\Documents\\teste dia 19_09\\coleta_lado_esquerdo.csv"
    ARQUIVO_TREINO_DIREITA = "C:\\Users\\User\\Documents\\teste dia 19_09\\coleta_lado_direito.csv"
    ARQUIVO_PARA_CLASSIFICAR = "C:\\Users\\User\\Documents\\teste dia 19_09\\coleta_mista.csv"
    #    ↑↑↑ EDITE OS TRÊS CAMINHOS ACIMA COM SEUS ARQUIVOS ↑↑↑

    # 3. Inicializar o pipeline com os 3 arquivos
    pipeline = EEGPipeline(
        caminho_esquerda=ARQUIVO_TREINO_ESQUERDA,
        caminho_direita=ARQUIVO_TREINO_DIREITA,
        caminho_classificacao=ARQUIVO_PARA_CLASSIFICAR,
        modelo="LDA"
    )
    
    # 4. Treinar o modelo
    pipeline.treinar()
    
    # 5. Iniciar a classificação no arquivo separado
    pipeline.iniciar_classificacao()