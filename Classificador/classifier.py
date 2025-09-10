# classifier.py

"""
Encapsula o pipeline de classificação (CSP + Classificador).
Permite treinar, avaliar, salvar, carregar e prever.
"""

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib

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