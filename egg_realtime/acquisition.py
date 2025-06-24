import sys
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from config import STREAM_NAME, PLOT_SEC, FS

class LSLStreamHandler:
    def __init__(self):
        print("Procurando stream EEG LSL...")
        streams = resolve_byprop('name', STREAM_NAME)
        if not streams:
            print("Nenhum stream EEG encontrado.")
            sys.exit()

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.n_channels = info.channel_count()

        # Obter nomes dos canais
        desc = info.desc()
        channels = desc.child('channels').child('channel')
        self.ch_names = []
        for _ in range(self.n_channels):
            self.ch_names.append(channels.child_value('label'))
            channels = channels.next_sibling()

        print(f"Conectado ao stream com {self.n_channels} canais:", self.ch_names)

        # Buffer circular
        self.buf_len = FS * PLOT_SEC
        self.buf = np.zeros((self.n_channels, self.buf_len))

    def update_buffer(self):
        sample, _ = self.inlet.pull_sample(timeout=0.0)
        if sample:
            self.buf = np.roll(self.buf, -1, axis=1)
            self.buf[:, -1] = sample[:self.n_channels]

    def get_channel_data(self, index, win_samps):
        return self.buf[index, -win_samps:]
