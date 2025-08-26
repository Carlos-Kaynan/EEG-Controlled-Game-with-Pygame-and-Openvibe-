import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Parâmetros ===
arquivo_csv = 'record-[2025.07.02-16.06.25].csv'
fs = 250  # Frequência de amostragem em Hz

# === Leitura dos dados ===
df = pd.read_csv(arquivo_csv)

# === Identifica os nomes dos canais EEG (ajuste se necessário) ===
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

    # Criar a figura com 5 subplots (um para cada banda)
    fig, axs = plt.subplots(len(bandas), 1, figsize=(10, 12))
    fig.suptitle(f"Espectro por Banda - Canal {canal}", fontsize=14, weight="bold")

    # Loop pelas bandas para gerar os gráficos individuais
    for i, (nome, (fmin, fmax)) in enumerate(bandas.items()):
        ax = axs[i]
        mask = (freq >= fmin) & (freq <= fmax)
        pot_total = np.sum(espectro[mask])

        # Plot do espectro somente naquela banda
        ax.plot(freq[mask], espectro[mask], color="blue", linewidth=1.2)
        ax.set_title(f"{nome} | Potência: {pot_total:.2f} µV²")
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Potência (µV²/Hz)")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # === Exibir potências no terminal ===
    print(f"\nPotências por banda no canal {canal}:")
    for nome, (fmin, fmax) in bandas.items():
        mask = (freq >= fmin) & (freq <= fmax)
        print(f"  {nome}: {np.sum(espectro[mask]):.2f} µV²")
