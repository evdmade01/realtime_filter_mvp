"""
realtime_filter_mvp_optimized.py

Optimized MVP-structured realtime signal/filter demo.
- Minimal dependencies (no TensorFlow, PyTorch, OpenCV, etc.)
- Improved filter response plotting (dB scale, better axes)
- Faster startup time

Dependencies (minimal):
  - PyQt6
  - pyqtgraph
  - numpy
  - scipy

Run:
  python realtime_filter_mvp_optimized.py
"""
import sys
import enum
import numpy as np
from scipy import signal
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
from typing import Tuple

# ----------------------
# MODEL
# ----------------------
class WaveType(enum.Enum):
    BLOCK = "Block"
    SAW = "Sawtooth"
    TRIANGLE = "Triangle"

class Model:
    def __init__(self, sample_rate=44100, buffer_size=4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        # parameters (defaults)
        self.input_freq = 440.0
        self.wave_type = WaveType.SAW
        self.filter_order = 1
        # For bandpass: lowcut & highcut in Hz
        self.band_low = 300.0
        self.band_high = 3000.0
        # For low/high: cutoff freq
        self.cutoff = 1000.0
        # window / buffer time vector
        self.t = np.arange(self.buffer_size) / self.sample_rate

    def generate_input(self) -> np.ndarray:
        f = self.input_freq
        t = self.t
        wt = 2 * np.pi * f * t
        if self.wave_type == WaveType.SAW:
            # sawtooth from scipy
            return signal.sawtooth(wt)
        elif self.wave_type == WaveType.TRIANGLE:
            return signal.sawtooth(wt, width=0.5)  # triangle
        elif self.wave_type == WaveType.BLOCK:
            # square wave 50% duty
            return signal.square(wt)
        else:
            return np.sin(wt)

    def design_filters_freq_response(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return frequency axis and frequency responses (magnitude in dB) for
        lowpass, highpass, bandpass designed with current parameters and order.
        """
        # Create frequency axis - more points for smoother curves
        N_freq = 2048  # More frequency points for smoother display
        freqs = np.logspace(0, np.log10(self.sample_rate/2), N_freq)  # Log-spaced from 1 Hz to Nyquist
        
        # Normalized frequencies for butter (0..1, where 1 is Nyquist)
        nyq = 0.5 * self.sample_rate

        # lowpass
        norm_cut = self.cutoff / nyq
        norm_cut = np.clip(norm_cut, 1e-6, 0.9999)
        b_lp, a_lp = signal.butter(self.filter_order, norm_cut, btype='low', analog=False)
        w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=freqs * 2.0 * np.pi / self.sample_rate)

        # highpass
        b_hp, a_hp = signal.butter(self.filter_order, norm_cut, btype='high', analog=False)
        w_hp, h_hp = signal.freqz(b_hp, a_hp, worN=freqs * 2.0 * np.pi / self.sample_rate)

        # bandpass
        low = np.clip(self.band_low / nyq, 1e-6, 0.9999)
        high = np.clip(self.band_high / nyq, 1e-6, 0.9999)
        if low >= high:
            high = min(low + 0.01, 0.9999)
        b_bp, a_bp = signal.butter(self.filter_order, [low, high], btype='band', analog=False)
        w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=freqs * 2.0 * np.pi / self.sample_rate)

        # Convert to dB
        h_lp_db = 20 * np.log10(np.abs(h_lp) + 1e-12)  # Add small value to avoid log(0)
        h_hp_db = 20 * np.log10(np.abs(h_hp) + 1e-12)
        h_bp_db = 20 * np.log10(np.abs(h_bp) + 1e-12)

        return freqs, h_lp_db, h_hp_db, h_bp_db

    def apply_filter_via_fft(self, input_sig: np.ndarray, filter_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple frequency-domain filtering:
         - compute FFT of input (real -> rfft)
         - compute theoretical filter frequency response on bins
         - multiply and IFFT back
        filter_type: 'low', 'high', 'band'
        returns: (output_time_domain, output_magnitude_spectrum_dB)
        """
        N = len(input_sig)
        freqs = np.fft.rfftfreq(N, d=1.0/self.sample_rate)
        # design filter response for these freqs
        nyq = 0.5 * self.sample_rate
        # choose analog/butter normalized
        if filter_type == 'low' or filter_type == 'high':
            cutoff = self.cutoff
            norm = cutoff / nyq
            b, a = signal.butter(self.filter_order, np.clip(norm, 1e-6, 0.9999),
                                 btype='low' if filter_type == 'low' else 'high')
        elif filter_type == 'band':
            low = np.clip(self.band_low / nyq, 1e-6, 0.9999)
            high = np.clip(self.band_high / nyq, 1e-6, 0.9999)
            if low >= high:
                high = min(low + 0.01, 0.9999)
            b, a = signal.butter(self.filter_order, [low, high], btype='band')
        else:
            # passthrough
            return input_sig, 20 * np.log10(np.abs(np.fft.rfft(input_sig)) + 1e-12)

        # get frequency response for those discrete frequencies
        w, h = signal.freqz(b, a, worN=freqs * 2.0 * np.pi / self.sample_rate)
        H = np.abs(h)

        # FFT of input
        X = np.fft.rfft(input_sig)
        Y = X * H
        y = np.fft.irfft(Y, n=N)
        
        # Convert output spectrum to dB
        Y_db = 20 * np.log10(np.abs(Y) + 1e-12)
        
        return y, Y_db


# ----------------------
# VIEW
# ----------------------
class View(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Filter MVP - Optimized")
        self.resize(1200, 900)

        # central widget + layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout()
        w.setLayout(layout)

        # Top controls area
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        # Wave selection
        self.wave_combo = QtWidgets.QComboBox()
        for wt in WaveType:
            self.wave_combo.addItem(wt.value)
        controls.addWidget(QtWidgets.QLabel("Input Wave:"))
        controls.addWidget(self.wave_combo)

        # Input frequency
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(1.0, 10000.0)
        self.freq_spin.setValue(440.0)
        self.freq_spin.setDecimals(2)
        controls.addWidget(QtWidgets.QLabel("Input Freq (Hz):"))
        controls.addWidget(self.freq_spin)

        # Filter order
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(1)
        controls.addWidget(QtWidgets.QLabel("Filter order:"))
        controls.addWidget(self.order_spin)

        # cutoff for low/high or midpoint for band
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(1.0, 20000.0)
        self.cutoff_spin.setValue(1000.0)
        controls.addWidget(QtWidgets.QLabel("Cutoff/Band Low (Hz):"))
        controls.addWidget(self.cutoff_spin)

        # Band high
        self.bandhigh_spin = QtWidgets.QDoubleSpinBox()
        self.bandhigh_spin.setRange(1.0, 20000.0)
        self.bandhigh_spin.setValue(3000.0)
        controls.addWidget(QtWidgets.QLabel("Band high (Hz):"))
        controls.addWidget(self.bandhigh_spin)

        # choose which PASS filter is applied to output (low/high/band)
        self.output_filter_combo = QtWidgets.QComboBox()
        self.output_filter_combo.addItems(["Low-pass", "High-pass", "Band-pass"])
        controls.addWidget(QtWidgets.QLabel("Output Filter:"))
        controls.addWidget(self.output_filter_combo)

        # Realtime ON/OFF
        self.rt_button = QtWidgets.QPushButton("Start Realtime")
        self.rt_button.setCheckable(True)
        controls.addWidget(self.rt_button)

        # Plot area - three stacked screens
        # Use pyqtgraph GraphicsLayoutWidget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=1)

        # 1) Input time domain
        self.p1 = self.plot_widget.addPlot(row=0, col=0, title="Input Signal (time domain)")
        self.p1_curve = self.p1.plot(pen='y')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setLabel('bottom', 'Time (ms)')
        self.p1.showGrid(x=True, y=True, alpha=0.3)

        # 2) Filter response in freq domain - IMPROVED (shows only selected filter)
        self.plot_widget.nextRow()
        self.p2 = self.plot_widget.addPlot(row=1, col=0, title="Selected Filter Response (frequency domain)")
        
        # Initialize single filter response curve - will be updated based on selection
        self.p2_active = None  # Will hold the currently active filter curve
        
        # Improved frequency response plot settings
        self.p2.setLogMode(x=True, y=False)
        self.p2.setLabel('left', 'Magnitude (dB)')
        self.p2.setLabel('bottom', 'Frequency (Hz)')
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend with better positioning
        self.legend = self.p2.addLegend(offset=(10, 10))
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 200))  # Semi-transparent white background
        
        # Set better Y-axis range for dB
        self.p2.setYRange(-80, 10)  # Typical dB range for filter responses

        # 3) Output: show time domain and freq domain side-by-side in the bottom row
        self.plot_widget.nextRow()
        # left bottom: output time-domain
        self.p3_time = self.plot_widget.addPlot(row=2, col=0, title="Output Signal (time domain)")
        self.p3_time_curve = self.p3_time.plot(pen='c')
        self.p3_time.setLabel('left', 'Amplitude')
        self.p3_time.setLabel('bottom', 'Time (ms)')
        self.p3_time.showGrid(x=True, y=True, alpha=0.3)

        # right bottom: output frequency domain - IMPROVED
        self.p3_freq = self.plot_widget.addPlot(row=2, col=1, title="Output Spectrum (freq domain)")
        self.p3_freq_curve = self.p3_freq.plot(pen='m')
        self.p3_freq.setLabel('left', 'Magnitude (dB)')
        self.p3_freq.setLabel('bottom', 'Frequency (Hz)')
        self.p3_freq.setLogMode(x=True, y=False)
        self.p3_freq.showGrid(x=True, y=True, alpha=0.3)

        # status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def bind_controls(self, presenter):
        # connect signals to presenter handlers
        self.wave_combo.currentIndexChanged.connect(presenter.on_wave_changed)
        self.freq_spin.valueChanged.connect(presenter.on_input_freq_changed)
        self.order_spin.valueChanged.connect(presenter.on_order_changed)
        self.cutoff_spin.valueChanged.connect(presenter.on_cutoff_changed)
        self.bandhigh_spin.valueChanged.connect(presenter.on_band_high_changed)
        self.output_filter_combo.currentIndexChanged.connect(presenter.on_output_filter_changed)
        self.rt_button.toggled.connect(presenter.on_realtime_toggled)

    def update_input_plot(self, arr: np.ndarray, time_ms: np.ndarray):
        self.p1_curve.setData(time_ms, arr)

    def update_filter_plots(self, freqs: np.ndarray, h_lp_db: np.ndarray, h_hp_db: np.ndarray, h_bp_db: np.ndarray, filter_params: dict, selected_filter: str):
        # Show only the selected filter response - much cleaner visualization
        mask = freqs > 0.1  # Avoid very low frequencies on log scale
        
        # Get current filter parameters
        cutoff = filter_params['cutoff']
        band_low = filter_params['band_low'] 
        band_high = filter_params['band_high']
        order = filter_params['order']
        
        # Clear previous plot
        self.p2.clear()
        self.legend.clear() if hasattr(self, 'legend') else None
        
        # Select which filter to display based on the selected output filter
        if selected_filter == 'low':
            filter_data = h_lp_db[mask]
            filter_name = f'Low-pass Filter (fc={cutoff:.0f}Hz, order={order})'
            pen_color = pg.mkPen('#FF4444', width=3)  # Bright red
            
        elif selected_filter == 'high':
            filter_data = h_hp_db[mask]
            filter_name = f'High-pass Filter (fc={cutoff:.0f}Hz, order={order})'
            pen_color = pg.mkPen('#44FF44', width=3)  # Bright green
            
        elif selected_filter == 'band':
            filter_data = h_bp_db[mask]
            filter_name = f'Band-pass Filter ({band_low:.0f}-{band_high:.0f}Hz, order={order})'
            pen_color = pg.mkPen('#4444FF', width=3)  # Bright blue
        
        else:
            # Fallback to low-pass
            filter_data = h_lp_db[mask]
            filter_name = f'Low-pass Filter (fc={cutoff:.0f}Hz, order={order})'
            pen_color = pg.mkPen('#FF4444', width=3)
        
        # Plot only the selected filter with prominent styling
        self.p2_active = self.p2.plot(freqs[mask], filter_data, 
                                      pen=pen_color, 
                                      name=filter_name)
        
        # Add cutoff frequency markers for better visualization
        if selected_filter == 'low' or selected_filter == 'high':
            # Add vertical line at cutoff frequency
            cutoff_line = pg.InfiniteLine(pos=cutoff, angle=90, 
                                         pen=pg.mkPen('orange', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
            self.p2.addItem(cutoff_line)
            # Add text label
            text_item = pg.TextItem(f'fc = {cutoff:.0f}Hz', anchor=(0, 1))
            text_item.setPos(cutoff, -5)
            self.p2.addItem(text_item)
            
        elif selected_filter == 'band':
            # Add vertical lines at band edges
            low_line = pg.InfiniteLine(pos=band_low, angle=90, 
                                      pen=pg.mkPen('orange', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
            high_line = pg.InfiniteLine(pos=band_high, angle=90, 
                                       pen=pg.mkPen('orange', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
            self.p2.addItem(low_line)
            self.p2.addItem(high_line)
            # Add text labels
            low_text = pg.TextItem(f'f1 = {band_low:.0f}Hz', anchor=(0, 1))
            low_text.setPos(band_low, -5)
            high_text = pg.TextItem(f'f2 = {band_high:.0f}Hz', anchor=(0, 1))
            high_text.setPos(band_high, -5)
            self.p2.addItem(low_text)
            self.p2.addItem(high_text)

        # Improved auto-scaling - show relevant frequency range
        f_min = max(0.1, freqs[mask][0])
        f_max = min(freqs[-1], 20000)  # Limit to audible range
        self.p2.setXRange(np.log10(f_min), np.log10(f_max), padding=0.02)

    def update_output_plots(self, out_time: np.ndarray, time_ms: np.ndarray, out_spec_freqs: np.ndarray, out_spec_mag_db: np.ndarray):
        self.p3_time_curve.setData(time_ms, out_time)
        # freq axis for rfft - improved scaling
        freqs = out_spec_freqs
        mask = freqs > 0.1  # Avoid DC and very low frequencies
        
        self.p3_freq_curve.setData(freqs[mask], out_spec_mag_db[mask])
        
        # Better frequency range for output spectrum
        f_min = max(0.1, freqs[mask][0])
        f_max = min(freqs[-1], 10000)  # Focus on relevant range
        self.p3_freq.setXRange(np.log10(f_min), np.log10(f_max))

    def set_status(self, txt: str):
        self.status.showMessage(txt)

# ----------------------
# PRESENTER
# ----------------------
class Presenter:
    def __init__(self, model: Model, view: View, update_interval_ms=60):
        self.model = model
        self.view = view
        self.timer = QtCore.QTimer()
        self.timer.setInterval(update_interval_ms)
        self.timer.timeout.connect(self.on_timer)
        self.view.bind_controls(self)
        self.update_interval_ms = update_interval_ms

        # initialize view with model defaults
        self.view.freq_spin.setValue(self.model.input_freq)
        # map wave type
        idx = list(WaveType).index(self.model.wave_type)
        self.view.wave_combo.setCurrentIndex(idx)
        self.view.order_spin.setValue(self.model.filter_order)
        self.view.cutoff_spin.setValue(self.model.cutoff)
        self.view.bandhigh_spin.setValue(self.model.band_high)
        self.view.output_filter_combo.setCurrentIndex(0)
        self.view.set_status("Ready - Optimized Version")

    # control handlers (view -> presenter)
    def on_wave_changed(self, idx):
        wt = list(WaveType)[idx]
        self.model.wave_type = wt
        self.view.set_status(f"Wave set to {wt.value}")

    def on_input_freq_changed(self, val):
        self.model.input_freq = float(val)
        self.view.set_status(f"Input freq: {val:.2f} Hz")

    def on_order_changed(self, val):
        self.model.filter_order = int(val)
        self.view.set_status(f"Filter order: {val}")

    def on_cutoff_changed(self, val):
        self.model.cutoff = float(val)
        self.model.band_low = float(val)  # Update band low to match cutoff
        self.view.set_status(f"Cutoff set to {val:.2f} Hz")

    def on_band_high_changed(self, val):
        self.model.band_high = float(val)
        self.view.set_status(f"Band high: {val:.2f} Hz")

    def on_output_filter_changed(self, idx):
        mapping = {0: 'low', 1: 'high', 2: 'band'}
        self.current_output_filter = mapping.get(idx, 'low')
        self.view.set_status(f"Output filter: {mapping.get(idx)}")

    def on_realtime_toggled(self, checked):
        if checked:
            self.timer.start()
            self.view.rt_button.setText("Stop Realtime")
            self.view.set_status("Realtime started")
        else:
            self.timer.stop()
            self.view.rt_button.setText("Start Realtime")
            self.view.set_status("Realtime stopped")

    def on_timer(self):
        # generate input
        x = self.model.generate_input()
        # Create time axis in milliseconds
        time_ms = self.model.t * 1000  # Convert seconds to milliseconds
        self.view.update_input_plot(x, time_ms)

        # compute and update filter frequency responses (now in dB)
        freqs, h_lp_db, h_hp_db, h_bp_db = self.model.design_filters_freq_response()
        
        # Pass current filter parameters and selected filter type
        filter_params = {
            'cutoff': self.model.cutoff,
            'band_low': self.model.band_low,
            'band_high': self.model.band_high,
            'order': self.model.filter_order
        }
        
        # Get currently selected filter type
        idx = self.view.output_filter_combo.currentIndex()
        mapping = {0: 'low', 1: 'high', 2: 'band'}
        selected_filter = mapping.get(idx, 'low')
        
        self.view.update_filter_plots(freqs, h_lp_db, h_hp_db, h_bp_db, filter_params, selected_filter)

        # apply selected output filter via FFT method
        idx = self.view.output_filter_combo.currentIndex()
        mapping = {0: 'low', 1: 'high', 2: 'band'}
        filter_name = mapping.get(idx, 'low')
        y, Ymag_db = self.model.apply_filter_via_fft(x, filter_name)
        # compute frequency bins (for plotting)
        freqs_out = np.fft.rfftfreq(len(x), d=1.0/self.model.sample_rate)
        self.view.update_output_plots(y, time_ms, freqs_out, Ymag_db)

# ----------------------
# MAIN
# ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    model = Model(sample_rate=44100, buffer_size=4096)
    view = View()
    presenter = Presenter(model, view, update_interval_ms=60)  # ~16-17 fps
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()