import numpy as np
from pylsl import StreamInlet, resolve_byprop

class EEGStream:
    def __init__(self, config: dict):
        self.config = config
        print("Procurando stream EEG na rede...")
        streams = resolve_byprop('name', 'openvibeSignal')
        if not streams:
            raise RuntimeError("Nenhum stream EEG encontrado! Verifique o OpenViBE ou outro software de streaming.")
        
        self.inlet = StreamInlet(streams[0])
        
        info = self.inlet.info()
        channel_count = info.channel_count()
        
        if self.config["N_CANAIS"] != channel_count:
            print(f"⚠️ Aviso: O número de canais foi ajustado de {self.config['N_CANAIS']} para {channel_count} (detectado do stream).")
            self.config["N_CANAIS"] = channel_count
            
        print(f"✅ Stream EEG '{info.name()}' encontrado com {self.config['N_CANAIS']} canais!")

    def coletar_dados(self, duracao: int) -> np.ndarray:
        n_amostras_total = duracao * self.config["FS"]
        dados = np.zeros((n_amostras_total, self.config["N_CANAIS"]))
        
        amostras_coletadas = 0
        while amostras_coletadas < n_amostras_total:
            samples, _ = self.inlet.pull_chunk(timeout=1.5, max_samples=self.config["FS"])
            if samples:
                chunk_array = np.array(samples)
                n_samples_chunk = chunk_array.shape[0]
                fim = amostras_coletadas + n_samples_chunk
                dados[amostras_coletadas:fim, :] = chunk_array[:, :self.config["N_CANAIS"]]
                amostras_coletadas += n_samples_chunk
        
        return dados.T