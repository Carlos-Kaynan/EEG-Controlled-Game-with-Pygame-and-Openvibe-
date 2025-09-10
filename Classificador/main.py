# main.py

"""
Ponto de entrada principal da aplicação BCI.
Importa a configuração, instancia o pipeline e inicia o processo.
"""

from config import CONFIG
from pipeline import EEGPipeline

def main():
    """Função principal que inicia o pipeline."""
    try:
        pipeline = EEGPipeline(CONFIG)
        
        # --- LÓGICA SIMPLIFICADA: Treina e depois classifica ---
        print("Iniciando o protocolo de treinamento...")
        pipeline.treinar()
        
        print("\nCarregando modelo para iniciar a classificação online...")
        pipeline.classificador.carregar_modelo()
        pipeline.classificar_online()
        # --------------------------------------------------------
        
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()