import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from config import FS, LIM_MU

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, data)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

def classify(ch1_signal, ch2_signal, fs):
    mu_ch1 = band_power(bandpass_filter(ch1_signal, 8, 13, fs), fs)
    mu_ch2 = band_power(bandpass_filter(ch2_signal, 8, 13, fs), fs)

    mu_diff = mu_ch2 - mu_ch1

    if mu_diff > LIM_MU:
        return "🖐 Esquerda"
    elif mu_diff < -LIM_MU:
        return "✊ Direita"
    else:
        return "⏸ Neutro"
