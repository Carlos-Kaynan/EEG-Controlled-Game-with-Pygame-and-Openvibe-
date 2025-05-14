# Teste com simulador

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
    win = 2 * sf  # Define uma janela de 2 segundos para o cálculo da densidade espectral de potência (PSD)
    freqs, psd = welch(eeg_data, sf, nperseg=win, axis=0)  # Calcula a PSD para cada canal do EEG usando o método de Welch
    
    # Seleciona os índices das frequências que correspondem à banda Alpha (8-13 Hz)
    idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
    
    # Seleciona os índices das frequências que correspondem à banda Beta (13-30 Hz)
    idx_beta = np.logical_and(freqs >= 13, freqs <= 30)
    
    # Calcula a potência absoluta da banda Alpha usando integração numérica (Simpson)
    alpha_power = simpson(psd[idx_alpha, :], dx=freqs[1] - freqs[0], axis=0)
    
    # Calcula a potência absoluta da banda Beta
    beta_power = simpson(psd[idx_beta, :], dx=freqs[1] - freqs[0], axis=0)
    
    # Calcula a potência total no espectro para normalizar
    total_power = simpson(psd, dx=freqs[1] - freqs[0], axis=0)
    
    # Calcula a potência relativa de Alpha e Beta (relação entre potência da banda e total)
    alpha_rel_power = alpha_power / total_power
    beta_rel_power = beta_power / total_power
    
    return alpha_rel_power, beta_rel_power  # Retorna as potências relativas

# Exemplo de uso com dados simulados
eeg_data = np.random.rand(1024, 4)  # Gera dados simulados de EEG (1024 amostras, 4 canais)
sf = 512  # Define a taxa de amostragem como 512 Hz
alpha, beta = extract_alpha_beta(eeg_data, sf)  # Extrai as bandas alpha e beta

# Exibe os resultados no console
print("Potência relativa Alpha:", alpha)
print("Potência relativa Beta:", beta)


''' Teste na Prática com openvibe sem simulação (teste com EEG físico) '''

from pylsl import StreamInlet, resolve_byprop
import time

# Procurando stream pelo nome configurado no OpenViBE
print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')  # Procura por streams LSL com o nome 'openvibeSignal'

# Criando o inlet para acessar os dados do stream
inlet = StreamInlet(streams[0])

eeg_data = []  # Lista para armazenar os dados EEG coletados

print("Recebendo dados EEG...")
try:
    start_time = time.time()  # Marca o tempo de início da coleta
    while time.time() - start_time < 10:  # Coleta os dados por 10 segundos
        sample, timestamp = inlet.pull_sample()  # Puxa uma nova amostra do stream
        eeg_data.append(sample)  # Adiciona a amostra à lista de dados
        print(f"Timestamp: {timestamp}, Dados: {sample}")  # Exibe o timestamp e os dados coletados
except KeyboardInterrupt:
    print("Interrupção manual.")  # Caso o usuário interrompa com Ctrl+C
finally:
    print("Coleta finalizada.")  # Mensagem final após o término da coleta
    print(f"Total de amostras coletadas: {len(eeg_data)}")  # Exibe o total de amostras armazenadas
