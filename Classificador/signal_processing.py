# signal_processing.py

"""
Funções de utilidade para o pré-processamento de sinais EEG.
Inclui filtros passa-banda, notch e remoção de artefatos.
"""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from typing import Tuple

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