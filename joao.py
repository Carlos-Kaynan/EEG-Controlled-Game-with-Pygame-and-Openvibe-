import sys
import random
import threading
import time
import platform
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from pylsl import StreamInlet, resolve_byprop
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import pygame
import pyttsx3
from queue import Queue, Empty


speak_queue = Queue()
IS_WINDOWS = platform.system().lower().startswith("win")

def tts_worker():
    """
    Worker único de TTS.
    Reinicia o engine a cada fala para evitar travamentos após a primeira execução.
    """
    while True:
        try:
            text = speak_queue.get(timeout=0.1)
        except Empty:
            continue

        if text is None:
            break  # caso queira encerrar no futuro

        engine = None
        try:
            # Em Windows, força sapi5; em outros, deixa default
            if IS_WINDOWS:
                engine = pyttsx3.init(driverName="sapi5")
            else:
                engine = pyttsx3.init()

            engine.setProperty("rate", 150)

            # tenta voz feminina, se existir
            try:
                voices = engine.getProperty("voices")
                chosen = None
                for v in voices:
                    # heurística simples
                    name = (v.name or "").lower()
                    if "female" in name or "fem" in name or "mulher" in name:
                        chosen = v.id
                        break
                if chosen is not None:
                    engine.setProperty("voice", chosen)
            except Exception:
                pass

            print(f"[TTS] Falando: {text}")
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
        except Exception as e:
            print(f"[TTS] Erro ao falar: {e}")
        finally:
            try:
                del engine
            except Exception:
                pass

def speak_async(text: str):
    """Enfileira a fala para o worker único (reinicializado por fala)."""
    speak_queue.put(text)

# inicia o worker de TTS (daemon)
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# ---------------------------
#   Parâmetros de EEG/BCI
# ---------------------------
fs = 512
win_sec = 1
win_samps = fs * win_sec
plot_sec = 2
buf_len = fs * plot_sec
lim_mu = 0.75

# --- PROGRESS BAR ---
progress_max = 100
progress = 0
last_pred = "none"
stimulus_direction = "none"
lock = threading.Lock()

# ---------------------------
#   Thread de estímulos
# ---------------------------
def run_pygame_stimulus():
    global stimulus_direction, progress

    pygame.init()
    screen = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("BCI Stimulus")
    font = pygame.font.Font(None, 36)

    right_img = pygame.image.load("C:/Users/User/Downloads/arrow_right.png")
    right_img = pygame.transform.scale(right_img, (100, 100))
    left_img = pygame.transform.flip(right_img, True, False)

    clock = pygame.time.Clock()
    arrow_timer = 0
    arrow_showing = False

    while progress < progress_max:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))

        now = pygame.time.get_ticks()
        # a cada 10s, mostra nova direção por 4s
        if now - arrow_timer >= 10000:
            arrow_showing = True
            with lock:
                stimulus_direction = random.choice(["left", "right"])
                if stimulus_direction == "left":
                    speak_async("esquerda")
                else:
                    speak_async("direita")
            arrow_timer = now
        elif now - arrow_timer >= 4000:
            arrow_showing = False

        # cruz verde (ajuste posições conforme queira)
        pygame.draw.line(screen, (0, 255, 0), (275, 136), (275, 156), 2)
        pygame.draw.line(screen, (0, 255, 0), (265, 146), (285, 146), 2)

        # seta
        if arrow_showing:
            with lock:
                if stimulus_direction == "left":
                    screen.blit(left_img, (250 - 100, 100))
                elif stimulus_direction == "right":
                    screen.blit(right_img, (250 + 50, 100))

        # barra de progresso
        pygame.draw.rect(screen, (255, 255, 255), (450, 100, 20, 150), 2)
        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (
                450,
                250 - int((progress / progress_max) * 150),
                20,
                int((progress / progress_max) * 150),
            ),
        )

        pygame.display.flip()
        clock.tick(60)

    screen.fill((0, 0, 0))
    text = font.render("SUCCESS!", True, (0, 255, 0))
    screen.blit(text, (220, 130))
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    QtWidgets.QApplication.instance().quit()

# ---------------------------
#  DSP básico (mantido)
# ---------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, data)

def band_power(signal, fs):
    freqs, psd = welch(signal, fs)
    return np.sum(psd)

# ---------------------------
#   Conexão LSL
# ---------------------------
print("Procurando stream EEG LSL...")
streams = resolve_byprop('name', 'openvibeSignal')
if not streams:
    print("Nenhum stream EEG encontrado.")
    sys.exit()

inlet = StreamInlet(streams[0])
info = inlet.info()
n_channels = info.channel_count()

# labels de canais
desc = info.desc()
channels = desc.child('channels').child('channel')
ch_names = []
for i in range(n_channels):
    ch_names.append(channels.child_value('label'))
    channels = channels.next_sibling()

try:
    ch1_index = ch_names.index('11')  # ajuste conforme seu mapeamento
    ch2_index = ch_names.index('14')
except ValueError:
    print("Canais '11' e '14' não encontrados no stream!")
    sys.exit()

# ---------------------------
#   Buffer & Janela gráfica
# ---------------------------
buf = np.zeros((n_channels, buf_len))
time_buf = np.linspace(-plot_sec, 0, buf_len)

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG - Visualização em Tempo Real")
win.resize(1600, 900)

plot_channels = [ch1_index, ch2_index]
plots, curves = [], []
for row, ch_idx in enumerate(plot_channels):
    p = win.addPlot(row=row, col=0, title=f"Canal {ch_names[ch_idx]} (idx {ch_idx})")
    p.setYRange(-100, 100)
    curve = p.plot(pen='w', width=2)
    plots.append(p)
    curves.append(curve)

# --- Text overlays for dominant band (per channel) ---
text_items = {}  # maps "ch1"/"ch2" -> pg.TextItem
for ch_key, p in zip(["ch1", "ch2"], plots):
    label = pg.TextItem(text="Banda: NONE", anchor=(0, 0), color=(230, 230, 230))
    p.addItem(label, ignoreBounds=True)  # keep inside plot; we'll position each update
    text_items[ch_key] = label

# ---------------------------
#   FFT + Bandas (cores)
# ---------------------------
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "mu":    (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
band_order = ["delta", "theta", "alpha", "mu", "beta", "gamma"]

band_colors = {
    "delta": (163, 73, 164),
    "theta": (120, 200, 255),
    "alpha": (80, 220, 150),
    "mu":    (80,  220, 150),
    "beta":  (255, 200, 100),
    "gamma": (237, 28, 36),
    "none":  (200, 200, 200),
}

win_fft_sec = 2.0
win_fft = int(fs * win_fft_sec)
hop = 64
fft_hann = np.hanning(win_fft)

ema_alpha = 0.3
consec_needed = 3
margin_dom = 0.15

state = {
    "ch1": {
        "ema": {k: 0.0 for k in bands},       # normalized bandpower EMA (0..1)
        "abs_bp": {k: 0.0 for k in bands},    # NEW: latest absolute bandpower
        "dom": "none", "streak": 0,
        "color": band_colors["none"], "target": band_colors["none"], "tween": 0
    },
    "ch2": {
        "ema": {k: 0.0 for k in bands},
        "abs_bp": {k: 0.0 for k in bands},    # NEW: latest absolute bandpower
        "dom": "none", "streak": 0,
        "color": band_colors["none"], "target": band_colors["none"], "tween": 0
    },
}
tween_steps = 10

def compute_psd(signal, fs):
    x = signal - np.mean(signal)
    if len(x) < win_fft:
        pad = win_fft - len(x)
        x = np.pad(x, (pad, 0))
    else:
        x = x[-win_fft:]
    x = x * fft_hann
    spec = np.fft.rfft(x)
    psd = (np.abs(spec) ** 2) / (np.sum(fft_hann**2) * fs)
    freqs = np.fft.rfftfreq(win_fft, d=1/fs)
    return freqs, psd

def bandpower_from_psd(freqs, psd, f_lo, f_hi):
    idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
    if len(idx) == 0:
        return 0.0
    return psd[idx].sum()

def normalize_bandpowers(bpowers):
    total = sum(bpowers.values()) + 1e-12
    return {k: v / total for k, v in bpowers.items()}

def ema_update(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def argmax_with_priority(dct, order):
    if not dct:
        return "none"
    maxv = max(dct.values())
    cands = [k for k, v in dct.items() if np.isclose(v, maxv)]
    for k in order:
        if k in cands:
            return k
    return cands[0]

def lerp_color(c0, c1, t, steps):
    if steps <= 1:
        return c1
    r = int(c0[0] + (c1[0]-c0[0]) * (t/steps))
    g = int(c0[1] + (c1[1]-c0[1]) * (t/steps))
    b = int(c0[2] + (c1[2]-c0[2]) * (t/steps))
    return (r, g, b)

# ---------------------------
#   Loop de atualização
# ---------------------------
def update():
    global buf, progress, last_pred

    # ingestão
    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample:
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = sample[:n_channels]

    # classificador original (mu_diff) e FFT/cores a cada 'hop'
    if update.counter % hop == 0:
        ch1_sig = buf[ch1_index, -win_samps:]
        ch2_sig = buf[ch2_index, -win_samps:]
        mu_diff = band_power(bandpass_filter(ch2_sig, 8, 13, fs), fs) - band_power(bandpass_filter(ch1_sig, 8, 13, fs), fs)

        pred = "none"
        if mu_diff > lim_mu:
            pred = "left"
        elif mu_diff < -lim_mu:
            pred = "right"

        with lock:
            expected = stimulus_direction

        if expected != "none" and pred == expected:
            progress += 1
            print(f"✅ Match: {pred} | Progress: {progress}/{progress_max}")
        else:
            print(f"❌ Mismatch or idle: Predicted={pred}, Expected={expected}")

        # PSD + bandas (2s)
        seg1 = buf[ch1_index, -win_fft:]
        seg2 = buf[ch2_index, -win_fft:]

        f1, psd1 = compute_psd(seg1, fs)
        f2, psd2 = compute_psd(seg2, fs)

        bp1 = {k: bandpower_from_psd(f1, psd1, *bands[k]) for k in bands}  # absolute bandpower
        bp2 = {k: bandpower_from_psd(f2, psd2, *bands[k]) for k in bands}

        # store latest absolute bandpowers (for label use between hops)
        state["ch1"]["abs_bp"] = bp1
        state["ch2"]["abs_bp"] = bp2

        nbp1 = normalize_bandpowers(bp1)
        nbp2 = normalize_bandpowers(bp2)

        for k in bands:
            state["ch1"]["ema"][k] = ema_update(state["ch1"]["ema"][k], nbp1[k], ema_alpha)
            state["ch2"]["ema"][k] = ema_update(state["ch2"]["ema"][k], nbp2[k], ema_alpha)

        for ch_key in ["ch1", "ch2"]:
            ema_dict = state[ch_key]["ema"]
            sorted_bands = sorted(ema_dict.items(), key=lambda kv: kv[1], reverse=True)
            dom_band = argmax_with_priority(ema_dict, band_order)

            if len(sorted_bands) > 1:
                first, second = sorted_bands[0][1], sorted_bands[1][1]
                margin_ok = (first - second) >= margin_dom
            else:
                margin_ok = True

            if margin_ok and dom_band != state[ch_key]["dom"]:
                state[ch_key]["streak"] += 1
                if state[ch_key]["streak"] >= consec_needed:
                    state[ch_key]["dom"] = dom_band
                    state[ch_key]["streak"] = 0
                    state[ch_key]["tween"] = 0
                    state[ch_key]["target"] = band_colors.get(dom_band, band_colors["none"])
            else:
                state[ch_key]["streak"] = 0

    # update plots + cores (with labels)
    for curve, ch_idx, ch_key, p in zip(curves, plot_channels, ["ch1", "ch2"], plots):
        curve.setData(time_buf, buf[ch_idx])

        st = state[ch_key]
        if st["tween"] < tween_steps:
            new_color = lerp_color(st["color"], st["target"], st["tween"], tween_steps)
            st["tween"] += 1
            st["color"] = new_color
        else:
            new_color = st["target"]
            st["color"] = new_color

        r, g, b = new_color
        curve.setPen(pg.mkPen((r, g, b), width=2))

        # --- Update label text (dominant band) ---
        ema = st["ema"]
        dom = st["dom"]
        abs_bp_dict = st["abs_bp"]  # latest absolute bandpowers

        if dom != "none":
            pct = int(max(0.0, min(1.0, ema.get(dom, 0.0))) * 100)
            f_lo, f_hi = bands[dom]
            abs_val = abs_bp_dict.get(dom, 0.0)
            # NOTE: units depend on input scaling; treat as "power (a.u.)" unless you know uV scaling.
            label_text = f"Banda: {dom.upper()} ({f_lo}-{f_hi} Hz)  {pct}% | P={abs_val:.2e} µV²"
        else:
            label_text = "Banda: NONE"

        lbl = text_items[ch_key]
        lbl.setText(label_text)

        # --- Position label at top-left (in data coords), robust to resizes/zoom ---
        (x0, x1), (y0, y1) = p.vb.viewRange()
        dx, dy = (x1 - x0), (y1 - y0)
        lbl.setPos(x0 + 0.02 * dx, y1 - 0.08 * dy)

    update.counter += 1

update.counter = 0
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000/fs))

# ---------------------------
#   Inicia Pygame (thread)
# ---------------------------
stim_thread = threading.Thread(target=run_pygame_stimulus)
stim_thread.start()

# ---------------------------
#   Execução GUI Qt
# ---------------------------
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()