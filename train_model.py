import pandas as pd
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === Carregar dataset CSV ===
df = pd.read_csv('1°Treinamento_Prototipo/eeg_abrir.csv.csv')

# Ajuste conforme seus canais
canais = df.columns[6:20]
sinal = df[canais].values
labels = df["label"].values  # 0 = esquerda, 1 = direita

# Reshape: (n_trials, n_channels, n_samples)
# Aqui você deve já ter seus trials cortados
X = sinal.reshape(-1, len(canais), 250)  # ex: 1s de janela (250 Hz)
y = labels

# === Treinar CSP + LDA ===
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
lda = LDA()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

# Salvar modelo
joblib.dump(clf, "csp_lda_model.joblib")
