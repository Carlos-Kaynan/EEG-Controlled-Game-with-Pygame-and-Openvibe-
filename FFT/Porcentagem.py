import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class EEGAnalyzer:
    def __init__(self, arquivo_csv: str, fs: int = 250):

        if not os.path.exists(arquivo_csv):
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_csv}")

        self.arquivo_csv = arquivo_csv
        self.fs = fs
        self.df = pd.read_csv(arquivo_csv)

        self.canais_eeg: list[str] = self.df.columns[6:20].tolist()
        print("Canais detectados:", self.canais_eeg)

        self.bandas: dict[str, tuple[float, float]] = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Teta (4-8 Hz)": (4, 8),
            "Alfa (8-13 Hz)": (8, 13),
            "Beta (13-30 Hz)": (13, 30),
            "Gama (30-45 Hz)": (30, 45)
        }

    def normalizar(self, sinal: np.ndarray) -> np.ndarray:
        return (sinal - np.mean(sinal)) / np.std(sinal)

    def filtrar(self, sinal: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        nyq = 0.5 * self.fs
        low, high = fmin / nyq, fmax / nyq
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, sinal)

    def segmentar(self, sinal: np.ndarray, janela: int) -> list[np.ndarray]:
        """
        Segmenta o sinal em janelas (epochs).

        Args:
            sinal (np.ndarray): Vetor de EEG.
            janela (int): Tamanho da janela em amostras.
        """
        return [sinal[i:i+janela] for i in range(0, len(sinal), janela) if len(sinal[i:i+janela]) == janela]


    def aplicar_fft(self, sinal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aplica FFT e retorna frequências e espectro."""
        N = len(sinal)
        freq = np.fft.rfftfreq(N, d=1/self.fs)
        espectro = np.abs(np.fft.rfft(sinal))**2
        return freq, espectro

    def calcular_potencias(self, freq: np.ndarray, espectro: np.ndarray) -> dict[str, float]:
        """Calcula potência absoluta em cada banda EEG."""
        potencias = {}
        for nome, (fmin, fmax) in self.bandas.items():
            mask = (freq >= fmin) & (freq <= fmax)
            potencias[nome] = float(np.sum(espectro[mask]))
        return potencias

    def calcular_potencias_relativas(self, potencias: dict[str, float]) -> dict[str, float]:
        """Calcula potência relativa (%) de cada banda."""
        total = sum(potencias.values())
        return {b: (p / total * 100) if total > 0 else 0 for b, p in potencias.items()}

    def plotar_bandas(self, canal: str) -> None:
        """Plota espectros por banda de um canal."""
        sinal = self.df[canal].dropna().values
        freq, espectro = self.aplicar_fft(sinal)

        fig, axs = plt.subplots(len(self.bandas), 1, figsize=(10, 12))
        fig.suptitle(f"Espectro por Banda - Canal {canal}", fontsize=14, weight="bold")

        for i, (nome, (fmin, fmax)) in enumerate(self.bandas.items()):
            ax = axs[i]
            mask = (freq >= fmin) & (freq <= fmax)
            pot_total = np.sum(espectro[mask])

            ax.plot(freq[mask], espectro[mask], color="blue", linewidth=1.2)
            ax.set_title(f"{nome} | Potência: {pot_total:.2f} µV²")
            ax.set_xlabel("Frequência (Hz)")
            ax.set_ylabel("Potência (µV²/Hz)")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plotar_heatmap(self) -> None:
        """Plota heatmap de potências médias por canal/banda."""
        resultados = []
        for canal in self.canais_eeg:
            sinal = self.df[canal].dropna().values
            freq, espectro = self.aplicar_fft(sinal)
            potencias = self.calcular_potencias(freq, espectro)
            resultados.append(list(potencias.values()))

        matriz = np.array(resultados)

        plt.figure(figsize=(10, 6))
        plt.imshow(matriz, cmap="viridis", aspect="auto")
        plt.colorbar(label="Potência (µV²)")
        plt.xticks(range(len(self.bandas)), list(self.bandas.keys()), rotation=45)
        plt.yticks(range(len(self.canais_eeg)), self.canais_eeg)
        plt.title("Heatmap de Potências por Canal e Banda")
        plt.tight_layout()
        plt.show()

    # ============================
    # === Execução ===
    # ============================
    def analisar_canal(self, canal: str) -> None:
        """Executa análise completa para um canal específico."""
        sinal = self.df[canal].dropna().values
        freq, espectro = self.aplicar_fft(sinal)
        potencias = self.calcular_potencias(freq, espectro)
        potencias_rel = self.calcular_potencias_relativas(potencias)

        print(f"\nCanal {canal} - Potências Absolutas:")
        for b, v in potencias.items():
            print(f"  {b}: {v:.2f} µV²")

        print(f"\nCanal {canal} - Potências Relativas (%):")
        for b, v in potencias_rel.items():
            print(f"  {b}: {v:.2f}%")

        self.plotar_bandas(canal)

    def analisar_todos_canais(self) -> None:
        """Executa análise em todos os canais e gera heatmap."""
        for canal in self.canais_eeg:
            self.analisar_canal(canal)
        self.plotar_heatmap()


# ============================
# === Execução principal ===
# ============================
if __name__ == "__main__":
    arquivo_csv = 'Coletas/record-[2025.07.02-16.06.25].csv'  # ajuste nome se necessário
    analisador = EEGAnalyzer(arquivo_csv, fs=250)
    analisador.analisar_todos_canais()
