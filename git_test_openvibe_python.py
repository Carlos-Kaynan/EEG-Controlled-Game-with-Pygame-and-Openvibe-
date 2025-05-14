from pylsl import StreamInlet, resolve_byprop  # Importa funções da biblioteca LSL para comunicação com fluxos de dados
import time
import numpy as np  # Biblioteca para manipulação numérica
import matplotlib.pyplot as plt  # Biblioteca para geração de gráficos
from scipy.integrate import simpson  # Método de integração numérica

# Início da aquisição de dados EEG via LSL (Lab Streaming Layer)
print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')  # Procura por fluxos com o nome 'openvibeSignal'

# Criação do objeto de entrada para leitura dos dados do stream
inlet = StreamInlet(streams[0])
count = 0
arr = []

# Coleta de 0.5 segundos de dados a 512 Hz (256 amostras)
for i in range(0, int(0.5 * 512)):
    sample, timestamp = inlet.pull_sample()  # Coleta uma amostra do stream
    arr.append(sample)  # Adiciona a amostra à lista
    count += 1
print('length', len(arr))  # Mostra o número total de amostras coletadas

# Função para extrair os dados de um canal específico
def channel(chnum):
    cha = []  # Lista para armazenar os dados do canal
    for i in range(len(arr)):
        cha.append(arr[i][chnum])  # Extrai o valor do canal chnum de cada amostra
    print("Tamanho do canal", chnum, ":", len(cha))
    return cha  # Retorna os dados do canal

# Teste da função com o canal 0
ch_data = channel(0)
print("Dados do Canal 0:", ch_data[:10])  # Mostra os primeiros 10 dados

# Função para plotar os dados de um único canal
def singlechannelgraph(sf, chdata, chno):
    sf = 512
    rate = sf
    time = np.arange(256) / sf  # Cria o eixo do tempo

    # Geração do gráfico
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(time, chdata, lw=1.5, color='k')  # Plota o sinal
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Voltagem')
    plt.xlim([time.min(), time.max()])
    plt.title('Canal %d EEG data' % (chno))

# Função para calcular a densidade espectral de potência (PSD)
def singlechannelPSD(channeldata, sf):
    from scipy import signal
    win = 0.5 * sf  # Janela de 0.5 segundos
    freqs, psd = signal.welch(channeldata, sf, nperseg=win)  # Welch PSD
    p = np.fft.rfft(channeldata)  # FFT (transformada rápida de Fourier)
    f = np.linspace(0, 512 / 2, len(p))  # Frequências correspondentes à FFT
    return freqs, psd, f, p

# Função para obter os índices das bandas específicas (ex: delta, alpha)
def Bandspecs_getidx_delta(lowb, highb, freqs, psd):
    low, high = lowb, highb
    idx_delta = np.logical_and(freqs >= low, freqs <= high)  # Índices da faixa desejada

    # Gráfico destacando a banda de interesse
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Densidade espectral de potência (uV^2 / Hz)')
    plt.xlim([0, 40])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Periodograma de Welch")
    return idx_delta

# Função para calcular a potência absoluta na banda delta (ou outra)
def deltapower(idx_delta1, freqs, psd1, f, p):
    freq_res = freqs[1] - freqs[0]  # Resolução em frequência do PSD
    fr_res = f[1] - f[0]  # Resolução da FFT
    delta_power = simpson(p[idx_delta1], dx=fr_res)  # Integração da área da banda
    total_power = simpson(p, dx=fr_res)  # Potência total
    delta_rel_power = delta_power / total_power  # Potência relativa (não usada aqui)
    return delta_power

# Função para calcular a potência relativa
def relpower(idx_delta, freqs, psd, f, p):
    freq_res = freqs[1] - freqs[0]
    fr_res = f[1] - f[0]
    delta_power = simpson(p[idx_delta], dx=fr_res)
    total_power = simpson(p, dx=fr_res)
    delta_rel_power = delta_power / total_power
    return delta_rel_power

# Função principal para executar a análise dos canais
def _main(nc, sf):
    nc = 4  # Número de canais
    sf = 512  # Frequência de amostragem
    chan = []

    # Coleta os dados de cada canal
    for m in range(0, nc):
        chan.append(channel(m))

    # Plota os sinais dos canais
    for n in range(0, nc):
        singlechannelgraph(sf, chan[n], n)

    freqs_all = []
    psd_all = []
    f_all = []
    p_all = []

    # Calcula a FFT e PSD para cada canal
    for o in range(0, nc):
        freqs, psd, f, p = singlechannelPSD(chan[o], sf)
        freqs_all.append(freqs)
        psd_all.append(psd)
        f_all.append(f)
        p_all.append(p)

    idx_delta = []
    # Define bandas de frequência de interesse
    idx_delta.append(Bandspecs_getidx_delta(4, 7, freqs_all[0], psd_all[0]))    # Banda theta
    idx_delta.append(Bandspecs_getidx_delta(8, 13, freqs_all[1], psd_all[1]))   # Banda alpha
    idx_delta.append(Bandspecs_getidx_delta(13, 30, freqs_all[2], psd_all[2]))  # Banda beta

    abspower_sec = []  # Potência absoluta
    for p1 in range(0, nc):
        abspower_prim = []
        for q in range(0, 3):
            abspower_prim.append((deltapower(idx_delta[q], freqs_all[p1], psd_all[p1], f_all[p1], p_all[p1])).real)
        abspower_sec.append(abspower_prim)

    relpower_sec = []  # Potência relativa
    for p1 in range(0, nc):
        relpower_prim = []
        for q in range(0, 3):
            relpower_prim.append((relpower(idx_delta[q], freqs_all[p1], psd_all[p1], f_all[p1], p_all[p1])).real)
        relpower_sec.append(relpower_prim)

    print(f'Print do idx_delta{idx_delta}')
    return relpower_sec  # Retorna a potência relativa para cada canal e banda

# Execução do script principal
nc = 4    # Número de canais
sf = 512  # Frequência de amostragem

print("Executando _main()...")
resultado = _main(nc, sf)

print("Resultado da análise:", resultado)  # Exibe os resultados da potência relativa

'''
# Versão interativa para entrada manual de parâmetros
nc,sf=input().split()
nc=int(nc)
sf=int(sf)
re_power=_main(nc,sf)
np_rev_power=np.array(re_power)  # Converte para matriz NumPy (CANAIS x 3 bandas)
print('---Matrix---')
print(np_rev_power)
print(np_rev_power.shape)
'''
