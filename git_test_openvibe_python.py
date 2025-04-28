from pylsl import StreamInlet, resolve_byprop
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

#testar as funçoes comentadas

print("Procurando stream LSL...")
streams = resolve_byprop('name', 'openvibeSignal')  

# Criando o inlet
inlet = StreamInlet(streams[0])
count=0
arr=[]
for i in range(0,int(0.5*512)):
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    #print(timestamp, sample,count)
    arr.append(sample)
    count=count+1
print('length', len(arr))



def channel(chnum):
    cha = []  # Lista para armazenar os dados do canal
    for i in range(len(arr)):  # Percorre os dados coletados
        cha.append(arr[i][chnum])  # Adiciona os dados do canal específico

    print("Tamanho do canal", chnum, ":", len(cha))  # Imprime o tamanho corretamente
    return cha  # Retorna os dados do canal

# Testando a função para o canal 0
ch_data = channel(0)  # Chama corretamente com um número de canal
print("Dados do Canal 0:", ch_data[:10])  # Exibe os primeiros 10 valores

def singlechannelgraph(sf,chdata,chno):
    sf = 512
    rate=sf
    time = np.arange(256) / sf

# Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(time, chdata, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.xlim([time.min(), time.max()])
    plt.title('Channel %d EEG data'%(chno))
def singlechannelPSD(channeldata,sf):

    from scipy import signal

# Define window length (0.5 seconds)
    win = 0.5 * sf
    freqs, psd = signal.welch(channeldata, sf, nperseg=win)
    p = (np.fft.rfft(channeldata))
   
    f = np.linspace(0, 512/2, len(p))
   # print(freqs)
    #print(f)
    #plt.plot(f,p)
# Plot the power spectrum
#sns.set(font_scale=1.2, style='white')
#    plt.figure(figsize=(8, 4))
#    plt.plot(freqs, psd, color='k', lw=2)
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Power spectral density (V^2 / Hz)')
#    plt.ylim([0, psd.max() * 1.1])
#    plt.title("Welch's periodogram")
#    plt.xlim([0, freqs.max()])
    return freqs,psd,f,p
#sns.despine()
def Bandspecs_getidx_delta(lowb,highb,freqs,psd):
    low, high = lowb, highb

# Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)

# Plot the power spectral density and fill the delta area
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([0, 40])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    return idx_delta
#sns.despine()

# Frequency resolution

def deltapower(idx_delta1,freqs,psd1,f,p):
    freq_res = freqs[1] - freqs[0]  # = 1 / 0.5 = 2
    #print(f)
    fr_res = f[1] - f[0]
# Compute the absolute power by approximating the area under the curve
    delta_power = simpson(p[idx_delta1], dx=fr_res)
    #print('Absolute delta power: %.3f uV^2' % delta_power)
    total_power = simpson(p, dx=fr_res)
    delta_rel_power = delta_power / total_power
    
    #print('Relative delta power: %.3f' % delta_rel_power)
    return delta_power
def relpower(idx_delta,freqs,psd,f,p):
    freq_res = freqs[1] - freqs[0]  # = 1 / 0.5 = 2
    fr_res = f[1] - f[0]
# Compute the absolute power by approximating the area under the curve
    delta_power = simpson(p[idx_delta], dx=fr_res)
    #print('Absolute delta power: %.3f uV^2' % delta_power)
    total_power = simpson(p, dx=fr_res)
    delta_rel_power = delta_power / total_power
    #print('Relative delta power: %.3f' % delta_rel_power)
    return delta_rel_power
def _main(nc,sf):
    nc=4
    sf=512
    chan=[]
    for m in range(0,nc):
        chan.append(channel(m))
    for n in range(0,nc):
        singlechannelgraph(sf,chan[n],n)
    
    freqs_all=[]
    psd_all=[]
    f_all=[]
    p_all=[]
    for o in range(0,nc):
        freqs,psd,f,p=singlechannelPSD(chan[o],sf)
        freqs_all.append(freqs)
        psd_all.append(psd)
        f_all.append(f)
        p_all.append(p)
    idx_delta=[]
#for theta 4-7hz
    idx_delta.append(Bandspecs_getidx_delta(4,7,freqs_all[0],psd_all[0]))
#for alpha 8-13hz
    idx_delta.append(Bandspecs_getidx_delta(8,13,freqs_all[1],psd_all[1]))
#for beta 13-30hz
    idx_delta.append(Bandspecs_getidx_delta(13,30,freqs_all[2],psd_all[2]))
    abspower_sec=[]
    for p1 in range(0,nc):
        abspower_prim=[]
        for q in range(0,3):
            abspower_prim.append((deltapower(idx_delta[q],freqs_all[p1],psd_all[p1],f_all[p1],p_all[p1])).real)    
        abspower_sec.append(abspower_prim)
    #print (len(abspower_sec),'x',int(len(abspower_prim)))
    relpower_sec=[]
    for p1 in range(0,nc):
        relpower_prim=[]
        for q in range(0,3):
            relpower_prim.append((relpower(idx_delta[q],freqs_all[p1],psd_all[p1],f_all[p1],p_all[p1])).real
                                )
        #print('insideq',relpower_prim)
        
        relpower_sec.append(relpower_prim)
    #print(relpower_sec)
    print(f'Print do idx_delta{idx_delta}')
    #print('inside-----p')



    return relpower_sec

nc = 4    # Número de canais (ajuste conforme necessário)
sf = 512  # Frequência de amostragem (Hz)

print("Executando _main()...")
resultado = _main(nc, sf)  # Chama a função

print("Resultado da análise:", resultado)  # Exibe os resultados



'''nc,sf=input().split()
nc=int(nc)
sf=int(sf)
re_power=_main(nc,sf)
np_rev_power=np.array(re_power)#----#CHNLx3 Matrix
print('---Matrix---')
print(np_rev_power)
print(np_rev_power.shape)'''
