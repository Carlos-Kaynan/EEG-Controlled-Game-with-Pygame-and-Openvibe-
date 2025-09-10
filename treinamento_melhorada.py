import numpy as np
import time
import matplotlib.pyplot as plt 
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from scipy.signal import butter, lfilter, iirnotch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from typing import List, Tuple

# ================================================================
# === CENTRAL DE CONFIGURAÇÕES ===
# ================================================================
CONFIG = {
    # Parâmetros do EEG (N_CANAIS será detectado automaticamente)
    "FS": 250,  # Taxa de amostragem (Hz)
    "N_CANAIS": 14, # Valor padrão, será sobrescrito pela detecção automática
    "CANAIS_EOG": [], # Desabilitado por padrão. Ex: [0, 13] se tiver canais EOG

    # Parâmetros de Filtros
    "LOWCUT": 8.0,  # Frequência de corte inferior para passa-banda
    "HIGHCUT": 30.0, # Frequência de corte superior para passa-banda
    "NOTCH_FREQ": 60.0, # Frequência do filtro notch (rede elétrica)
    "NOTCH_Q": 30,

    # Parâmetros de Treinamento
    "DURACAO_TAREFA": 30, # Duração de cada tarefa de imaginação (em segundos)
    "DURACAO_DESCANSO": 10, # Duração do descanso entre tarefas
    "JANELA_S": 1.0, # Tamanho da janela de época em segundos
    "JANELA_AMOSTRAS": int(250 * 1.0), # Tamanho da janela em amostras

    # Parâmetros do Modelo
    "MODELO": "LDA",  # "LDA" ou "SVM"
    "CSP_COMPONENTES": 4, # Número de componentes para o CSP
    "NOME_ARQUIVO_MODELO": "modelo_bci_motor.pkl",
}


# ================================================================
# === FILTROS DE PRÉ-PROCESSAMENTO ===
# ================================================================
def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Cria os coeficientes para um filtro Butterworth passa-banda."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def aplicar_bandpass(sinal: np.ndarray, config: dict) -> np.ndarray:
    """Aplica um filtro passa-banda ao sinal EEG."""
    b, a = butter_bandpass(config["LOWCUT"], config["HIGHCUT"], config["FS"])
    return lfilter(b, a, sinal, axis=1)

def aplicar_notch(sinal: np.ndarray, config: dict) -> np.ndarray:
    """Aplica um filtro notch para remover ruído da rede elétrica."""
    b, a = iirnotch(w0=config["NOTCH_FREQ"] / (config["FS"] / 2), Q=config["NOTCH_Q"])
    return lfilter(b, a, sinal, axis=1)

def remover_artefatos_olhos(sinal: np.ndarray, config: dict) -> np.ndarray:
    """Remove artefatos de piscadas usando regressão linear simples."""
    if not config["CANAIS_EOG"]:
        return sinal
    sinal_limpo = sinal.copy()
    for canal_eog_idx in config["CANAIS_EOG"]:
        if canal_eog_idx < sinal.shape[0]: # Verifica se o canal EOG existe
            eog = sinal[canal_eog_idx, :]
            ganho = np.dot(sinal, eog) / np.dot(eog, eog)
            sinal_limpo -= np.outer(ganho, eog)
    return sinal_limpo


# ================================================================
# === CLASSE DE COLETA DO LSL (COM DETECÇÃO AUTOMÁTICA) ===
# ================================================================
class EEGStream:
    """Gerencia a conexão e coleta de dados de um stream EEG LSL."""
    def __init__(self, config: dict):
        self.config = config
        print("🔍 Procurando stream EEG na rede...")
        streams = resolve_byprop('name', 'openvibeSignal')
        if not streams:
            raise RuntimeError("❌ Nenhum stream EEG encontrado! Verifique o OpenViBE ou outro software de streaming.")
        
        self.inlet = StreamInlet(streams[0])
        
        info = self.inlet.info()
        channel_count = info.channel_count()
        
        if self.config["N_CANAIS"] != channel_count:
            print(f"⚠️ Aviso: O número de canais foi ajustado de {self.config['N_CANAIS']} para {channel_count} (detectado do stream).")
            self.config["N_CANAIS"] = channel_count
            
        print(f"✅ Stream EEG '{info.name()}' encontrado com {self.config['N_CANAIS']} canais!")

    def coletar_dados(self, duracao: int) -> np.ndarray:
        """Coleta dados por uma duração específica em segundos."""
        n_amostras_total = duracao * self.config["FS"]
        dados = np.zeros((n_amostras_total, self.config["N_CANAIS"]))
        
        amostras_coletadas = 0
        while amostras_coletadas < n_amostras_total:
            samples, _ = self.inlet.pull_chunk(timeout=1.5, max_samples=self.config["FS"])
            if samples:
                chunk_array = np.array(samples)
                n_samples_chunk = chunk_array.shape[0]
                fim = amostras_coletadas + n_samples_chunk
                dados[amostras_coletadas:fim, :] = chunk_array[:, :self.config["N_CANAIS"]]
                amostras_coletadas += n_samples_chunk
        
        return dados.T


# ================================================================
# === CLASSE DE PRÉ-PROCESSAMENTO ===
# ================================================================
class EEGPreprocessador:
    """Responsável por criar épocas a partir do sinal contínuo."""
    def __init__(self, config: dict):
        self.config = config

    def criar_epocas(self, sinal: np.ndarray, classe: int) -> Tuple[List[np.ndarray], List[int]]:
        """Divide o sinal contínuo em épocas não sobrepostas."""
        X, y = [], []
        janela = self.config["JANELA_AMOSTRAS"]
        n_amostras = sinal.shape[1]
        for i in range(0, n_amostras - janela, janela):
            epoca = sinal[:, i:i + janela]
            X.append(epoca)
            y.append(classe)
        return X, y


# ================================================================
# === CLASSE DE CLASSIFICAÇÃO ===
# ================================================================
class EEGClassificador:
    """Encapsula o pipeline de classificação (CSP + Classificador)."""
    def __init__(self, config: dict):
        self.config = config
        self.csp = CSP(n_components=config["CSP_COMPONENTES"], reg=None, log=True, norm_trace=False)
        
        if config["MODELO"].upper() == "SVM":
            self.clf = Pipeline([("CSP", self.csp), ("SVM", SVC(kernel="rbf", C=2, gamma="scale"))])
        else:
            self.clf = Pipeline([("CSP", self.csp), ("LDA", LDA())])

    def treinar(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo com validação cruzada e exibe os resultados."""
        print("\n🔄 Avaliando classificador com cross-validation...")
        scores = cross_val_score(self.clf, X, y, cv=5, scoring="accuracy")
        print(f"📊 Acurácia média: {np.mean(scores)*100:.2f}% (std: {np.std(scores)*100:.2f}%)")
        
        print("💪 Treinando modelo final com todos os dados...")
        self.clf.fit(X, y)
        print("✅ Treinamento concluído!")

    def salvar_modelo(self):
        """Salva o modelo treinado em um arquivo."""
        nome_arquivo = self.config["NOME_ARQUIVO_MODELO"]
        joblib.dump(self.clf, nome_arquivo)
        print(f"💾 Modelo salvo em: {nome_arquivo}")

    def carregar_modelo(self):
        """Carrega um modelo pré-treinado de um arquivo."""
        nome_arquivo = self.config["NOME_ARQUIVO_MODELO"]
        self.clf = joblib.load(nome_arquivo)
        print(f"📂 Modelo carregado de: {nome_arquivo}")

    def prever(self, epoca: np.ndarray) -> int:
        """Prevê a classe de uma única época."""
        return self.clf.predict(epoca[np.newaxis, :, :])[0]

# ================================================================
# === VISUALIZADOR PARA FEEDBACK (AGORA NO CONSOLE) ===
# ================================================================
class VisualizadorConsole: # <-- NOVA CLASSE SEM GRÁFICOS
    """Classe para gerenciar o feedback visual para o usuário via console."""
    def mostrar_prompt(self, texto: str, cor: str = 'black'):
        """Exibe um texto no console."""
        print(f"==> {texto}")
        
    def fechar(self):
        """Método vazio para manter a compatibilidade."""
        pass

# ================================================================
# === PIPELINE COMPLETO ===
# ================================================================
class EEGPipeline:
    """Orquestra todo o processo de BCI, do treinamento à classificação."""
    def __init__(self, config: dict):
        self.config = config
        self.stream = EEGStream(self.config)
        self.preprocessador = EEGPreprocessador(self.config)
        self.classificador = EEGClassificador(self.config)
        self.visualizador = VisualizadorConsole() # <-- USA A NOVA CLASSE

    def _coletar_e_processar_tarefa(self, classe: int, texto_prompt: str) -> Tuple[List[np.ndarray], List[int]]:
        """Coleta, filtra e epocas os dados para uma única tarefa."""
        self.visualizador.mostrar_prompt("DESCANSE")
        time.sleep(self.config["DURACAO_DESCANSO"])
        
        self.visualizador.mostrar_prompt(texto_prompt)
        sinal_bruto = self.stream.coletar_dados(self.config["DURACAO_TAREFA"])
        
        sinal_filtrado = aplicar_notch(sinal_bruto, self.config)
        sinal_filtrado = aplicar_bandpass(sinal_filtrado, self.config)
        sinal_filtrado = remover_artefatos_olhos(sinal_filtrado, self.config)
        
        return self.preprocessador.criar_epocas(sinal_filtrado, classe)

    def treinar(self):
        """Executa a fase completa de treinamento."""
        print("\n=== 🧠 FASE DE TREINAMENTO 🧠 ===")
        
        X_e, y_e = self._coletar_e_processar_tarefa(0, "IMAGINE MÃO ESQUERDA")
        X_d, y_d = self._coletar_e_processar_tarefa(1, "IMAGINE MÃO DIREITA")

        self.visualizador.mostrar_prompt("Processando dados...")

        X = np.array(X_e + X_d)
        y = np.array(y_e + y_d)

        print(f"\n📊 Shape dos dados de treino: X={X.shape}, y={y.shape}")
        self.classificador.treinar(X, y)

        y_pred = self.classificador.clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        print("\n📌 Matriz de Confusão (dados de treino):")
        print(cm)
        print("\n📌 Relatório de Classificação (dados de treino):")
        print(classification_report(y, y_pred, target_names=["Esquerda", "Direita"]))

        self.classificador.salvar_modelo()

    def classificar_online(self):
        """Inicia a classificação em tempo real com feedback via console."""
        print("\n=== 🚀 CLASSIFICAÇÃO ONLINE 🚀 ===")
        print("Pressione Ctrl+C para sair.")
        self.visualizador.mostrar_prompt("Iniciando...")
        
        try:
            while True:
                samples, _ = self.stream.inlet.pull_chunk(
                    timeout=2.0, max_samples=self.config["JANELA_AMOSTRAS"]
                )
                if len(samples) < self.config["JANELA_AMOSTRAS"]:
                    continue
                
                epoca_bruta = np.array(samples).T[:self.config["N_CANAIS"], :]
                
                epoca_filtrada = aplicar_notch(epoca_bruta, self.config)
                epoca_filtrada = aplicar_bandpass(epoca_filtrada, self.config)
                epoca_filtrada = remover_artefatos_olhos(epoca_filtrada, self.config)
                
                predicao = self.classificador.prever(epoca_filtrada)
                
                if predicao == 0:
                    print("Predição: 🖐️ ESQUERDA")
                else:
                    print("Predição: DIREITA 🖐️")

        except KeyboardInterrupt:
            print("\nEncerrando a classificação online.")
        finally:
            self.visualizador.fechar()


# ================================================================
# === PONTO DE ENTRADA ===
# ================================================================
def main():
    """Função principal que inicia o pipeline."""
    try:
        pipeline = EEGPipeline(CONFIG)
        
        # --- LÓGICA SIMPLIFICADA: Treina e depois classifica ---
        print("Iniciando o protocolo de treinamento...")
        pipeline.treinar()
        
        print("\nCarregando modelo para iniciar a classificação online...")
        pipeline.classificador.carregar_modelo()
        pipeline.classificar_online()
        # --------------------------------------------------------
        
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    main()