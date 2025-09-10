# preprocessing.py

"""
Responsável por criar épocas (janelas de tempo) a partir do sinal contínuo.
"""

import numpy as np
from typing import List, Tuple

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