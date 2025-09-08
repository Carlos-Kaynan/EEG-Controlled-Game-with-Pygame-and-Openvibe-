import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EEGAnalyzer:
    def __init__(self, arquivo_csv, fs=250):
        self.arquivo_csv = arquivo_csv
        self.fs = fs
        self.df = pd.read_csv(arquivo_csv)
        self.canais_eeg = self.df.columns[6:20].tolist()
        print("Canais detectados:", self.canais_eeg)

        # Definição das bandas EEG
        self.bandas = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Teta (4-8 Hz)": (4, 8),
            "Alfa (8-13 Hz)": (8, 13),
            "Beta (13-30 Hz)": (13, 30),
            "Gama (30-45 Hz)": (30, 45)
        }

    def aplicar_fft(self, sinal):
        """Aplica FFT e retorna frequências e espectro de potência."""
        N = len(sinal)
        freq = np.fft.rfftfreq(N, d=1/self.fs)
        espectro = np.abs(np.fft.rfft(sinal))**2
        return freq, espectro

    def calcular_potencias(self, freq, espectro):
        """Calcula as potências por banda e suas porcentagens."""
        potencias = []
        nomes_bandas = list(self.bandas.keys())

        # Potência por banda
        for nome, (fmin, fmax) in self.bandas.items():
            mask = (freq >= fmin) & (freq <= fmax)
            potencia_total = np.sum(espectro[mask])
            potencias.append(potencia_total)

        # Cálculo das porcentagens
        potencia_total_geral = np.sum(potencias)
        porcentagens = [(p / potencia_total_geral) * 100 for p in potencias]

        return nomes_bandas, potencias, porcentagens

    def plotar_potencias(self, canal, nomes_bandas, potencias, porcentagens):
        """Plota gráfico de barras com potências e porcentagens."""
        plt.figure(figsize=(9, 5))
        cores = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
        barras = plt.bar(nomes_bandas, potencias, color=cores, alpha=0.8)

        # Adicionar valores absolutos e porcentagens acima das barras
        for barra, potencia, porcentagem in zip(barras, potencias, porcentagens):
            plt.text(barra.get_x() + barra.get_width()/2,
                     barra.get_height(),
                     f"{potencia:.1e}\n({porcentagem:.1f}%)",
                     ha='center', va='bottom', fontsize=9)

        plt.title(f"Potência por Banda EEG - Canal {canal}", fontsize=14, weight="bold")
        plt.xlabel("Bandas de Frequência")
        plt.ylabel("Potência Total (µV²)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def analisar(self):
        """Executa a análise de todos os canais EEG."""
        for canal in self.canais_eeg:
            sinal = self.df[canal].dropna().values
            freq, espectro = self.aplicar_fft(sinal)
            nomes_bandas, potencias, porcentagens = self.calcular_potencias(freq, espectro)

            # Exibir no terminal
            print(f"\n=== Potências no canal {canal} ===")
            for nome, potencia, porcentagem in zip(nomes_bandas, potencias, porcentagens):
                print(f"{nome}: {potencia:.2f} µV² ({porcentagem:.2f}%)")

            # Plotar gráfico
            self.plotar_potencias(canal, nomes_bandas, potencias, porcentagens)


# === Execução ===
if __name__ == "__main__":
    arquivo_csv = 'Coletas/record-[2025.07.02-16.06.25].csv'
    analisador = EEGAnalyzer(arquivo_csv, fs=250)
    analisador.analisar()
