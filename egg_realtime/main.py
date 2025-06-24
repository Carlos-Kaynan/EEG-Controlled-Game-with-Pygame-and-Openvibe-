from config import FS, WIN_SEC, TARGET_CHANNELS
from acquisition import LSLStreamHandler
from processing import classify
from interface import EEGVisualizer
from PyQt5 import QtCore

def main():
    lsl = LSLStreamHandler()
    vis = EEGVisualizer(lsl.n_channels, lsl.ch_names, lsl.buf_len)

    try:
        ch1_idx = lsl.ch_names.index(TARGET_CHANNELS[0])
        ch2_idx = lsl.ch_names.index(TARGET_CHANNELS[1])
    except ValueError:
        print(f"Canais {TARGET_CHANNELS} não encontrados no stream!")
        return

    win_samps = FS * WIN_SEC

    def update():
        lsl.update_buffer()

        if update.counter % win_samps == 0:
            ch1_data = lsl.get_channel_data(ch1_idx, win_samps)
            ch2_data = lsl.get_channel_data(ch2_idx, win_samps)

            estado = classify(ch1_data, ch2_data, FS)
            print(estado)
            vis.update_status(estado)

        vis.update_plot(lsl.buf)
        update.counter += 1

    update.counter = 0

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(1000 / FS))

    vis.start()

if __name__ == "__main__":
    main()
