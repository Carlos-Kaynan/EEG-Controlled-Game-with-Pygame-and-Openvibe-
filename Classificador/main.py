from config import CONFIG
from pipeline import EEGPipeline

def main():
    try:
        pipeline = EEGPipeline(CONFIG)
        
        print("Iniciando o protocolo de treinamento...")
        pipeline.treinar()
        
        print("\nCarregando modelo para iniciar a classificação online...")
        pipeline.classificador.carregar_modelo()
        pipeline.classificar_online()
        
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()