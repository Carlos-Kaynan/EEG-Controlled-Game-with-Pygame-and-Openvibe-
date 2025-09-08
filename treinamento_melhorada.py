import numpy as np
import time
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class EEGStream:
    """Classe para gerenciar a conexão e coleta de dados do LSL."""
    def __init__(self, n_canais: int = 14, fs: int = 250):
        self.n_canais = n_canais
        self.fs = fs

        print("🔍 Procurando stream EEG...")
        streams = resolve_stream("type", "EEG", timeout=5.0)
        if not streams:
            raise RuntimeError("❌ Nenhum stream EEG encontrado! Verifique o OpenViBE.")
        self.inlet = StreamInlet(streams[0])
        print("✅ Stream EEG encontrado!")

    def coletar_dados(self, classe: int, duracao: int) -> np.ndarray:
        """
        Coleta dados de EEG para uma classe específica.
        Usa pull_chunk() para melhor desempenho.
        """
        print(f"==> Inicie o movimento da mão {'ESQUERDA' if classe == 0 else 'DIREITA'} por {duracao}s")
        time.sleep(2)
        dados = []
        inicio = time.time()

        while (time.time() - inicio) < duracao:
            samples, _ = self.inlet.pull_chunk(timeout=1.0, max_samples=self.fs)
            if samples:
                dados.extend(np.array(samples)[:, :self.n_canais])

        return np.array(dados).T  # (n_canais, n_amostras)

    def coletar_amostra(self) -> np.ndarray:
        """Coleta uma única amostra."""
        sample, _ = self.inlet.pull_sample()
        return np.array(sample[:self.n_canais])


class EEGPreprocessador:
    """Classe para criar épocas a partir dos sinais brutos."""
    def __init__(self, janela: int = 250):
        self.janela = janela

    def criar_epocas(self, sinal: np.ndarray, classe: int):
        X, y = [], []
        n_amostras = sinal.shape[1]

        for i in range(0, n_amostras - self.janela, self.janela):
            epoca = sinal[:, i:i + self.janela]
            X.append(epoca)
            y.append(classe)

        return X, y


class EEGClassificador:
    """Classe para treinar e classificar sinais EEG usando CSP + LDA/SVM."""
    def __init__(self, n_componentes: int = 6, modelo: str = "LDA"):
        self.modelo = modelo.upper()
        self.csp = CSP(n_components=n_componentes, reg=None, log=True, norm_trace=False)

        if self.modelo == "SVM":
            self.clf = Pipeline([("CSP", self.csp), ("SVM", SVC(kernel="rbf", C=1))])
        else:
            self.clf = Pipeline([("CSP", self.csp), ("LDA", LDA())])

    def treinar(self, X: np.ndarray, y: np.ndarray):
        """Treina o classificador com validação cruzada."""
        print("\n🔄 Avaliando classificador com cross-validation...")
        scores = cross_val_score(self.clf, X, y, cv=5)
        print(f"📊 Acurácia média: {scores.mean()*100:.2f}%")

        print("\n🚀 Treinando classificador final...")
        self.clf.fit(X, y)
        print("✅ Treinamento concluído!")

    def prever(self, epoca: np.ndarray) -> int:
        """Classifica uma única época."""
        return self.clf.predict(epoca)[0]


class EEGPipeline:
    """Pipeline completo para treino e classificação online."""
    def __init__(self, fs=250, n_canais=14, janela=250, duracao_treino=30, modelo="LDA"):
        self.fs = fs
        self.n_canais = n_canais
        self.janela = janela
        self.duracao_treino = duracao_treino

        # Inicializa módulos
        self.stream = EEGStream(n_canais, fs)
        self.preprocessador = EEGPreprocessador(janela)
        self.classificador = EEGClassificador(modelo=modelo)
        self.predicoes = []  # para gráfico do histórico

    def treinar(self):
        """Fase de treinamento."""
        print("\n=== Fase de Treinamento ===")
        esquerda = self.stream.coletar_dados(0, self.duracao_treino)
        direita = self.stream.coletar_dados(1, self.duracao_treino)

        X_e, y_e = self.preprocessador.criar_epocas(esquerda, 0)
        X_d, y_d = self.preprocessador.criar_epocas(direita, 1)

        X = np.array(X_e + X_d)
        y = np.array(y_e + y_d)

        if np.isnan(X).any():
            raise ValueError("⚠️ Dados contêm NaN. Verifique o sinal EEG!")

        print(f"📊 Shape treino X: {X.shape}, y: {y.shape}")

        self.classificador.treinar(X, y)

    def classificar_online(self):
        """Classificação em tempo real com gráfico dinâmico."""
        print("\n=== Classificação Online ===")
        buffer = []

        # Configura gráfico em tempo real
        plt.ion()
        fig, (ax_sinal, ax_pred) = plt.subplots(2, 1, figsize=(10, 6))

        # Gráfico do sinal EEG (canal 1)
        ax_sinal.set_title("Sinal EEG - Canal 1")
        linha_sinal, = ax_sinal.plot(np.zeros(self.janela))
        ax_sinal.set_ylim([-200, 200])

        # Gráfico do histórico de predições
        ax_pred.set_title("Histórico de Predições")
        linha_pred, = ax_pred.plot([])
        ax_pred.set_ylim([-0.5, 1.5])

        while True:
            amostra = self.stream.coletar_amostra()
            buffer.append(amostra)

            # Mantém buffer deslizante
            if len(buffer) > self.janela:
                buffer.pop(0)

            # Atualiza gráfico do sinal
            sinal_canal1 = [b[0] for b in buffer]
            linha_sinal.set_ydata(sinal_canal1)
            linha_sinal.set_xdata(np.arange(len(buffer)))
            ax_sinal.set_xlim([0, len(buffer)])

            # Classificação com janela deslizante
            if len(buffer) >= self.janela:
                X_live = np.array(buffer[-self.janela:]).T[np.newaxis, :, :]
                pred = self.classificador.prever(X_live)
                self.predicoes.append(pred)

                # Atualiza histórico de predições
                linha_pred.set_ydata(self.predicoes)
                linha_pred.set_xdata(np.arange(len(self.predicoes)))
                ax_pred.set_xlim([0, len(self.predicoes)])

                print("🖐️ Esquerda" if pred == 0 else "🖐️ Direita")

            plt.pause(0.01)


if __name__ == "__main__":
    pipeline = EEGPipeline(modelo="LDA")  # Troque para "SVM" se quiser
    pipeline.treinar()
    pipeline.classificar_online()
