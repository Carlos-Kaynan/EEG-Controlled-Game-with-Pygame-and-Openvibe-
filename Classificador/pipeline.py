import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple
from lsl_stream import EEGStream
from preprocessing import EEGPreprocessador
from classifier import EEGClassificador
from visualization import VisualizadorConsole
import signal_processing as sp

class EEGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.stream = EEGStream(self.config)
        self.preprocessador = EEGPreprocessador(self.config)
        self.classificador = EEGClassificador(self.config)
        self.visualizador = VisualizadorConsole()

    def _coletar_e_processar_tarefa(self, classe: int, texto_prompt: str) -> Tuple[List[np.ndarray], List[int]]:
        self.visualizador.mostrar_prompt("DESCANSE")
        time.sleep(self.config["DURACAO_DESCANSO"])
        
        self.visualizador.mostrar_prompt(texto_prompt)
        sinal_bruto = self.stream.coletar_dados(self.config["DURACAO_TAREFA"])
        
        sinal_filtrado = sp.aplicar_notch(sinal_bruto, self.config)
        sinal_filtrado = sp.aplicar_bandpass(sinal_filtrado, self.config)
        sinal_filtrado = sp.remover_artefatos_olhos(sinal_filtrado, self.config)
        
        return self.preprocessador.criar_epocas(sinal_filtrado, classe)

    def treinar(self):
        print("\n===  FASE DE TREINAMENTO  ===")
        
        X_e, y_e = self._coletar_e_processar_tarefa(0, "IMAGINE MÃO ESQUERDA")
        X_d, y_d = self._coletar_e_processar_tarefa(1, "IMAGINE MÃO DIREITA")

        self.visualizador.mostrar_prompt("Processando dados...")

        X = np.array(X_e + X_d)
        y = np.array(y_e + y_d)

        print(f"\n📊 Shape dos dados de treino: X={X.shape}, y={y.shape}")
        self.classificador.treinar(X, y)

        y_pred = self.classificador.clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        print("\n📌 Matriz de Confusão (dados de treino):")
        print(cm)
        print("\n📌 Relatório de Classificação (dados de treino):")
        print(classification_report(y, y_pred, target_names=["Esquerda", "Direita"]))

        self.classificador.salvar_modelo()

    def classificar_online(self):
        print("\n===  CLASSIFICAÇÃO ONLINE  ===")
        print("Pressione Ctrl+C para sair.")
        self.visualizador.mostrar_prompt("Iniciando...")
        
        try:
            while True:
                samples, _ = self.stream.inlet.pull_chunk(
                    timeout=2.0, max_samples=self.config["JANELA_AMOSTRAS"]
                )
                if len(samples) < self.config["JANELA_AMOSTRAS"]:
                    continue
                
                epoca_bruta = np.array(samples).T[:self.config["N_CANAIS"], :]
                
                epoca_filtrada = sp.aplicar_notch(epoca_bruta, self.config)
                epoca_filtrada = sp.aplicar_bandpass(epoca_filtrada, self.config)
                epoca_filtrada = sp.remover_artefatos_olhos(epoca_filtrada, self.config)
                
                predicao = self.classificador.prever(epoca_filtrada)
                
                if predicao == 0:
                    print("Predição: ESQUERDA")
                else:
                    print("Predição: DIREITA")

        except KeyboardInterrupt:
            print("\nEncerrando a classificação online.")
        finally:
            self.visualizador.fechar()