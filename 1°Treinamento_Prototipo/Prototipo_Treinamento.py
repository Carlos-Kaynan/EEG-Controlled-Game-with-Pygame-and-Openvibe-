# -*- coding: utf-8 -*
# treinamento_motor_imagery.py
# Treina um classificador com 3 classes: abrir mão direita, fechar mão esquerda, e descanso
#edite o caminho do arquivo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.signal import butter, lfilter, welch
import os


# === Funções auxiliares ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

def extract_features(window, fs):
    c3 = window[:, 0]
    c4 = window[:, 1]

    mu_c3 = bandpass_filter(c3, 8, 13, fs)
    beta_c3 = bandpass_filter(c3, 13, 30, fs)

    mu_c4 = bandpass_filter(c4, 8, 13, fs)
    beta_c4 = bandpass_filter(c4, 13, 30, fs)

    def band_power(signal):
        freqs, psd = welch(signal, fs)
        return np.sum(psd)

    return [
        band_power(mu_c3),
        band_power(beta_c3),
        band_power(mu_c4),
        band_power(beta_c4)
    ]

# === Parâmetros ===
fs = 512
window_size = 1  # segundos
samples_per_window = fs * window_size

# === Arquivos CSV (ajuste os nomes se diferente) ===
arquivos = {
    "abrir": ("C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\eeg_abrir.csv.csv", 1),
    "fechar": ("C:\\Users\\carlo\\OneDrive\\Área de Trabalho\\eeg_fechar.csv.csv", 2),
    "neutro": ("c:\\Users\\carlo\\OneDrive\\Área de Trabalho\\eeg_neutro.csv.csv", 0)
}

X = []
y = []

for classe, (arquivo, rotulo) in arquivos.items():
    print(f"\nProcessando {arquivo} como classe {classe.upper()}...")
    df = pd.read_csv(arquivo)
    dados = df.iloc[:, 11:14].values  # C3 e C4
    print("Índice de C3:", df.columns.get_loc('10'))
    print("Índice de C4:", df.columns.get_loc('13'))

    


    total_janelas = len(dados) // int(samples_per_window)

    for i in range(total_janelas):
        janela = dados[i * int(samples_per_window):(i + 1) * int(samples_per_window)]
        feats = extract_features(janela, fs)
        X.append(feats)
        y.append(rotulo)

X = np.array(X)
y = np.array(y)

# === Treinamento ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
clf.fit(X_train, y_train)

print("\n=== Avaliação ===")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Neutro", "Abrir", "Fechar"]))

# === Salvando modelo treinado (opcional) ===
import joblib
os.makedirs("modelo", exist_ok=True)
joblib.dump(clf, "modelo/svm_motor_imagery.joblib")
print("\nModelo salvo em 'modelo/svm_motor_imagery.joblib'")


