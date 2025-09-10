import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from typing import Tuple

def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def aplicar_bandpass(sinal: np.ndarray, config: dict) -> np.ndarray:
    b, a = butter_bandpass(config["LOWCUT"], config["HIGHCUT"], config["FS"])
    return lfilter(b, a, sinal, axis=1)

def aplicar_notch(sinal: np.ndarray, config: dict) -> np.ndarray:
    b, a = iirnotch(w0=config["NOTCH_FREQ"] / (config["FS"] / 2), Q=config["NOTCH_Q"])
    return lfilter(b, a, sinal, axis=1)

def remover_artefatos_olhos(sinal: np.ndarray, config: dict) -> np.ndarray:
    if not config["CANAIS_EOG"]:
        return sinal
    sinal_limpo = sinal.copy()
    for canal_eog_idx in config["CANAIS_EOG"]:
        if canal_eog_idx < sinal.shape[0]:
            eog = sinal[canal_eog_idx, :]
            ganho = np.dot(sinal, eog) / np.dot(eog, eog)
            sinal_limpo -= np.outer(ganho, eog)
    return sinal_limpo