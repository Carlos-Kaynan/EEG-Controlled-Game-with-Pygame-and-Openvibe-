import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
O que esse código faz

° Mostra todas as bandas EEG no mesmo gráfico.
° Coloriza cada faixa de frequência para fácil interpretação.
° Calcula e exibe potência por banda para cada canal.
° Limita o gráfico até 45 Hz, que é a faixa mais usada em EEG clínico.

'''
# === Parâmetros ===
arquivo_csv = 'record-[2025.07.02-16.06.25].csv' #adicione o caminho onde o CSV está salvo
fs = 250  # Frequência de amostragem em Hz

# === Leitura dos dados ===
df = pd.read_csv(arquivo_csv)

# === Identifica os nomes dos canais EEG (ajuste o índice se necessário) ===
canais_eeg = df.columns[6:20].tolist()
print("Canais detectados:", canais_eeg)

# === Definir bandas EEG ===
bandas = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Teta (4-8 Hz)": (4, 8),
    "Alfa (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gama (30-45 Hz)": (30, 45)
}

# === Função para aplicar FFT ===
def aplicar_fft(sinal, fs):
    N = len(sinal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    espectro = np.abs(np.fft.rfft(sinal))**2
    return freq, espectro

# === Loop pelos canais ===
for canal in canais_eeg:
    sinal = df[canal].dropna().values
    freq, espectro = aplicar_fft(sinal, fs)

    # === Cálculo da potência por banda ===
    potencias = {}
    for nome, (fmin, fmax) in bandas.items():
        mask = (freq >= fmin) & (freq <= fmax)
        potencias[nome] = np.sum(espectro[mask])

    # === Plot ===
    plt.figure(figsize=(12, 5))
    plt.plot(freq, espectro, color="black", linewidth=1.2)

    # Destacar regiões de cada banda com cores
    cores = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
    for i, (nome, (fmin, fmax)) in enumerate(bandas.items()):
        plt.axvspan(fmin, fmax, color=cores[i], alpha=0.3, label=nome)

    plt.title(f"Espectro de Frequência - Canal {canal}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Potência (µV²/Hz)")
    plt.xlim(0, 45)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # === Exibir potências no terminal ===
    print(f"\nPotências por banda no canal {canal}:")
    for nome, valor in potencias.items():
        print(f"  {nome}: {valor:.2f} µV²")
