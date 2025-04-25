#Teste com simulador

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

def extract_alpha_beta(eeg_data, sf):
    """
    Extrai as bandas Alpha (8-13 Hz) e Beta (13-30 Hz) do sinal EEG.
    
    Parâmetros:
    - eeg_data: array numpy (formato [amostras, canais])
    - sf: taxa de amostragem em Hz
    
    Retorna:
    - alpha_power: Potência relativa da banda Alpha por canal
    - beta_power: Potência relativa da banda Beta por canal
    """
    win = 2 * sf  # Janela de 2 segundos para cálculo da PSD
    freqs, psd = welch(eeg_data, sf, nperseg=win, axis=0)  # Cálculo da PSD
    
    # Índices das bandas de interesse
    idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
    idx_beta = np.logical_and(freqs >= 13, freqs <= 30)
    
    # Cálculo da potência absoluta
    alpha_power = simpson(psd[idx_alpha, :], dx=freqs[1] - freqs[0], axis=0)
    beta_power = simpson(psd[idx_beta, :], dx=freqs[1] - freqs[0], axis=0)
    
    # Potência total
    total_power = simpson(psd, dx=freqs[1] - freqs[0], axis=0)
    
    # Cálculo da potência relativa
    alpha_rel_power = alpha_power / total_power
    beta_rel_power = beta_power / total_power
    
    return alpha_rel_power, beta_rel_power

# Exemplo de uso
eeg_data = np.random.rand(1024, 4)  # Simulação de EEG (1024 amostras, 4 canais)
sf = 512  # Taxa de amostragem em Hz
alpha, beta = extract_alpha_beta(eeg_data, sf)
print("Potência relativa Alpha:", alpha)
print("Potência relativa Beta:", beta)



#Teste na Prática com openvibe sem simulação


from pylsl import StreamInlet, resolve_byprop
import time

# Procurando stream pelo nome configurado no OpenViBE
print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')  # Nome exato definido no OpenViBE

# Criando o inlet
inlet = StreamInlet(streams[0])

eeg_data = []  # Lista para armazenar os dados

print("Recebendo dados EEG...")
try:
    start_time = time.time()
    while time.time() - start_time < 10:  # Coletando por 10 segundos
        sample, timestamp = inlet.pull_sample()
        eeg_data.append(sample)  # Armazenando em lista
        print(f"Timestamp: {timestamp}, Dados: {sample}")
except KeyboardInterrupt:
    print("Interrupção manual.")
finally:
    print("Coleta finalizada.")
    print(f"Total de amostras coletadas: {len(eeg_data)}")
