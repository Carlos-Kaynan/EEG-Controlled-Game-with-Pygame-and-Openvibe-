import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP

class EEGAnalyzer:
    def __init__(self, arquivo_csv, fs=250):
        self.arquivo_csv = arquivo_csv
        self.fs = fs
        self.df = pd.read_csv(arquivo_csv)
        self.canais_eeg = self.df.columns[6:20].tolist()
        print("Canais detectados:", self.canais_eeg)

    def preparar_epocas_manual(self, blocos, tamanho_janela=250):
        """
        Cria X e y a partir de blocos contínuos, dividindo em janelas menores.
        blocos = {
            0: [(inicio1, fim1), ...],  # mão esquerda
            1: [(inicio1, fim1), ...]   # mão direita
        }
        """
        X, y = [], []
        dados = self.df[self.canais_eeg].values

        for classe, lista_blocos in blocos.items():
            for inicio, fim in lista_blocos:
                # Dividir o bloco em janelas de tamanho_janela
                for start in range(inicio, fim, tamanho_janela):
                    end = start + tamanho_janela
                    if end > fim:
                        break
                    epoca = dados[start:end].T  # shape (n_canais, n_times)
                    X.append(epoca)
                    y.append(classe)

        X = np.array(X)
        y = np.array(y)
        print("Número de épocas:", X.shape[0])
        print("Formato X:", X.shape)
        print("Formato y:", y.shape)
        return X, y

    def classificar_csp_lda(self, blocos, tamanho_janela=250):
        """Aplica CSP + LDA usando blocos contínuos divididos em janelas."""
        X, y = self.preparar_epocas_manual(blocos, tamanho_janela)

        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        lda = LDA()
        clf = Pipeline([("CSP", csp), ("LDA", lda)])

        # Stratified K-Fold com 5 splits agora funciona porque teremos muitas épocas
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

        print("\n=== Classificação CSP + LDA ===")
        print("Acurácia média: %.2f%%" % (np.mean(scores) * 100))


# === Execução ===
if __name__ == "__main__":
    arquivo_csv = 'Coletas/record-[2025.07.02-16.06.25].csv'
    analisador = EEGAnalyzer(arquivo_csv, fs=250)

    # Defina os blocos contínuos em amostras
    blocos = {
        0: [(0, 2500), (5000, 7500)],    # mão esquerda
        1: [(2500, 5000), (7500, 10000)] # mão direita
    }

    # Executar CSP + LDA dividindo blocos em janelas de 1s (250 amostras)
    analisador.classificar_csp_lda(blocos, tamanho_janela=250)
