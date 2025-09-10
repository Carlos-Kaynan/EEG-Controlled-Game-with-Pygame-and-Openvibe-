# config.py

"""
Arquivo de configuração central para o projeto BCI.
Todos os parâmetros ajustáveis devem ser definidos aqui.
"""

CONFIG = {
    # Parâmetros do EEG (N_CANAIS será detectado automaticamente)
    "FS": 250,  # Taxa de amostragem (Hz)
    "N_CANAIS": 14, # Valor padrão, será sobrescrito pela detecção automática
    "CANAIS_EOG": [], # Desabilitado por padrão. Ex: [0, 13] se tiver canais EOG

    # Parâmetros de Filtros
    "LOWCUT": 8.0,  # Frequência de corte inferior para passa-banda
    "HIGHCUT": 30.0, # Frequência de corte superior para passa-banda
    "NOTCH_FREQ": 60.0, # Frequência do filtro notch (rede elétrica)
    "NOTCH_Q": 30,

    # Parâmetros de Treinamento
    "DURACAO_TAREFA": 30, # Duração de cada tarefa de imaginação (em segundos)
    "DURACAO_DESCANSO": 10, # Duração do descanso entre tarefas
    "JANELA_S": 1.0, # Tamanho da janela de época em segundos
    "JANELA_AMOSTRAS": int(250 * 1.0), # Tamanho da janela em amostras

    # Parâmetros do Modelo
    "MODELO": "LDA",  # "LDA" ou "SVM"
    "CSP_COMPONENTES": 4, # Número de componentes para o CSP
    "NOME_ARQUIVO_MODELO": "modelo_bci_motor.pkl",
}