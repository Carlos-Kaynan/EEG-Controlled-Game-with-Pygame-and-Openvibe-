import numpy as np
from typing import List, Tuple

class EEGPreprocessador:
    def __init__(self, config: dict):
        self.config = config

    def criar_epocas(self, sinal: np.ndarray, classe: int) -> Tuple[List[np.ndarray], List[int]]:
        X, y = [], []
        janela = self.config["JANELA_AMOSTRAS"]
        n_amostras = sinal.shape[1]
        for i in range(0, n_amostras - janela, janela):
            epoca = sinal[:, i:i + janela]
            X.append(epoca)
            y.append(classe)
        return X, y