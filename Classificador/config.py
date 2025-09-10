CONFIG = {
    "FS": 250,  # Taxa de amostragem (Hz)
    "N_CANAIS": 14,
    "CANAIS_EOG": [],

    # Parâmetros de Filtros
    "LOWCUT": 8.0,  # Frequência de corte inferior para passa-banda
    "HIGHCUT": 30.0, # Frequência de corte superior para passa-banda
    "NOTCH_FREQ": 60.0, # Frequência do filtro notch (rede elétrica)
    "NOTCH_Q": 30,

    # Parâmetros de Treinamento
    "DURACAO_TAREFA": 30, # Duração de cada tarefa de imaginação
    "DURACAO_DESCANSO": 10, # Duração do descanso entre tarefas
    "JANELA_S": 1.0, # Tamanho da janela de época em segundos
    "JANELA_AMOSTRAS": int(250 * 1.0), # Tamanho da janela em amostras

    # Parâmetros do Modelo
    "MODELO": "LDA",
    "CSP_COMPONENTES": 4, # Número de componentes para o CSP
    "NOME_ARQUIVO_MODELO": "modelo_bci_motor.pkl",
}