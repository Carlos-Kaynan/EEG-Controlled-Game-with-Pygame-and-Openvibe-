import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from config import FS, PLOT_SEC

class EEGVisualizer:
    def __init__(self, n_channels, ch_names, buffer_len):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="EEG - Visualização em Tempo Real")
        self.win.resize(1600, 900)

        self.time_buf = np.linspace(-PLOT_SEC, 0, buffer_len)
        self.plots = []
        self.curves = []

        for i in range(n_channels):
            p = self.win.addPlot(row=i // 4, col=i % 4, title=f"Canal {ch_names[i]}")
            p.setYRange(-100, 100)
            curve = p.plot(pen='w', width=1)
            self.plots.append(p)
            self.curves.append(curve)

        # Label de status
        self.status_label = pg.TextItem(text='Estado: ⏸', color='y', anchor=(0,1))
        self.win.addItem(self.status_label, row=n_channels // 4 + 1, col=0)

    def update_plot(self, buffer):
        for i, curve in enumerate(self.curves):
            curve.setData(self.time_buf, buffer[i])

    def update_status(self, status):
        self.status_label.setText(f'Estado: {status}')

    def start(self):
        QtWidgets.QApplication.instance().exec_()
