"""
realtime_filter_lecture.py

Educational Realtime Filter Application for ENGR202
Based on First-Order Filter Theory (Oregon State University College of Engineering)

This application demonstrates:
1. First-order RC Low-pass and High-pass filters
2. Transfer function H(s) analysis
3. Magnitude and Phase response
4. Time constant and cutoff frequency relationships
5. Step response analysis
6. Practical vs Theoretical comparisons

Educational Objectives:
- Understand transfer functions H(s) = Y(s)/X(s)
- Learn cutoff frequency fc = 1/(2πRC)
- Visualize magnitude |H(jω)| and phase ∠H(jω)
- Observe time domain vs frequency domain behavior

Author: Based on Oregon State University ENGR202 curriculum
Dependencies: PyQt6, pyqtgraph, numpy, scipy
"""

import sys
import numpy as np
from scipy import signal
from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from typing import Tuple, Dict
import math

# ----------------------
# THEORETICAL MODEL
# ----------------------
class FirstOrderFilter:
    """
    First-order filter model based on RC circuit analysis
    
    Low-pass:  H(s) = 1/(1 + sRC) = ωc/(s + ωc)
    High-pass: H(s) = sRC/(1 + sRC) = s/(s + ωc)
    
    Where: ωc = 2πfc = 1/RC (cutoff frequency in rad/s)
    """
    
    def __init__(self):
        # Circuit parameters
        self.R = 1000.0  # Resistance in Ohms
        self.C = 1e-6    # Capacitance in Farads
        self.filter_type = 'lowpass'  # 'lowpass' or 'highpass'
        
        # Derived parameters
        self.update_parameters()
    
    def update_parameters(self):
        """Update derived parameters when R or C changes"""
        self.tau = self.R * self.C  # Time constant τ = RC
        self.fc = 1.0 / (2 * np.pi * self.tau)  # Cutoff frequency fc = 1/(2πRC)
        self.wc = 2 * np.pi * self.fc  # Angular cutoff frequency ωc = 2πfc
    
    def transfer_function_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return transfer function coefficients for scipy.signal
        
        Low-pass:  H(s) = ωc/(s + ωc)  -> num=[ωc], den=[1, ωc]
        High-pass: H(s) = s/(s + ωc)   -> num=[1, 0], den=[1, ωc]
        """
        if self.filter_type == 'lowpass':
            num = np.array([self.wc])
            den = np.array([1.0, self.wc])
        else:  # highpass
            num = np.array([1.0, 0.0])
            den = np.array([1.0, self.wc])
        
        return num, den
    
    def frequency_response(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate theoretical frequency response |H(jω)| and ∠H(jω)
        
        Returns:
            magnitude_db: Magnitude in dB
            phase_deg: Phase in degrees
        """
        num, den = self.transfer_function_coefficients()
        w = 2 * np.pi * frequencies  # Convert Hz to rad/s
        
        # Calculate H(jω) using scipy
        h_complex = signal.freqs(num, den, w)[1]
        
        # Extract magnitude and phase
        magnitude_db = 20 * np.log10(np.abs(h_complex) + 1e-12)
        phase_deg = np.angle(h_complex, deg=True)
        
        return magnitude_db, phase_deg
    
    def theoretical_magnitude_db(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Analytical magnitude response in dB
        
        Low-pass:  |H(jω)| = 1/√(1 + (ω/ωc)²)
        High-pass: |H(jω)| = (ω/ωc)/√(1 + (ω/ωc)²)
        """
        w = 2 * np.pi * frequencies
        w_ratio = w / self.wc  # ω/ωc
        
        if self.filter_type == 'lowpass':
            magnitude = 1.0 / np.sqrt(1 + w_ratio**2)
        else:  # highpass
            magnitude = w_ratio / np.sqrt(1 + w_ratio**2)
        
        return 20 * np.log10(magnitude + 1e-12)
    
    def theoretical_phase_deg(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Analytical phase response in degrees
        
        Low-pass:  ∠H(jω) = -arctan(ω/ωc)
        High-pass: ∠H(jω) = 90° - arctan(ω/ωc)
        """
        w = 2 * np.pi * frequencies
        w_ratio = w / self.wc
        
        if self.filter_type == 'lowpass':
            phase_rad = -np.arctan(w_ratio)
        else:  # highpass
            phase_rad = np.pi/2 - np.arctan(w_ratio)
        
        return np.degrees(phase_rad)
    
    def step_response(self, time: np.ndarray) -> np.ndarray:
        """
        Theoretical step response
        
        Low-pass:  y(t) = 1 - e^(-t/τ)  for t ≥ 0
        High-pass: y(t) = e^(-t/τ)     for t ≥ 0
        """
        response = np.zeros_like(time)
        positive_time = time >= 0
        
        if self.filter_type == 'lowpass':
            # Avoid overflow for large negative time/tau ratios
            exp_arg = np.clip(-time[positive_time] / self.tau, -50, 50)
            response[positive_time] = 1.0 - np.exp(exp_arg)
        else:  # highpass
            exp_arg = np.clip(-time[positive_time] / self.tau, -50, 50)
            response[positive_time] = np.exp(exp_arg)
        
        return response


# ----------------------
# SIGNAL GENERATOR
# ----------------------
class SignalGenerator:
    """Educational signal generator for filter testing"""
    
    def __init__(self, sample_rate=44100, buffer_size=4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.t = np.arange(buffer_size) / sample_rate
        
        # Signal parameters
        self.signal_type = 'sine'  # 'sine', 'square', 'step', 'impulse'
        self.frequency = 100.0  # Hz
        self.amplitude = 1.0
    
    def generate_signal(self) -> np.ndarray:
        """Generate test signal based on current parameters"""
        if self.signal_type == 'sine':
            return self.amplitude * np.sin(2 * np.pi * self.frequency * self.t)
        
        elif self.signal_type == 'square':
            return self.amplitude * signal.square(2 * np.pi * self.frequency * self.t)
        
        elif self.signal_type == 'step':
            # Unit step at t = buffer_size/4
            step_time = len(self.t) // 4
            step_signal = np.zeros_like(self.t)
            step_signal[step_time:] = self.amplitude
            return step_signal
        
        elif self.signal_type == 'impulse':
            # Impulse at t = buffer_size/4
            impulse_signal = np.zeros_like(self.t)
            impulse_signal[len(self.t) // 4] = self.amplitude * self.sample_rate  # Scale for proper impulse
            return impulse_signal
        
        else:
            return np.zeros_like(self.t)


# ----------------------
# EDUCATIONAL VIEW
# ----------------------
class EducationalFilterView(QtWidgets.QMainWindow):
    """Educational interface for first-order filter analysis"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("First-Order Filter Analysis - ENGR202 Educational Tool")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Create main widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left panel: Controls and theory
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Plots
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
        # Status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - First-Order Filter Educational Tool")
    
    def create_left_panel(self) -> QtWidgets.QWidget:
        """Create control panel with theory and parameters"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(350)
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Theory section
        theory_group = QtWidgets.QGroupBox("First-Order Filter Theory")
        theory_layout = QtWidgets.QVBoxLayout(theory_group)
        
        # Add theoretical formulas
        formulas = QtWidgets.QTextEdit()
        formulas.setMaximumHeight(200)
        formulas.setHtml("""
        <h3>Transfer Functions:</h3>
        <b>Low-pass:</b> H(s) = ωc/(s + ωc)<br>
        <b>High-pass:</b> H(s) = s/(s + ωc)<br><br>
        
        <h3>Key Relationships:</h3>
        τ = RC (time constant)<br>
        fc = 1/(2πRC) (cutoff frequency)<br>
        ωc = 2πfc (angular frequency)<br><br>
        
        <h3>At cutoff frequency:</h3>
        |H(jωc)| = -3dB (70.7% of input)<br>
        ∠H(jωc) = ±45°
        """)
        formulas.setReadOnly(True)
        theory_layout.addWidget(formulas)
        layout.addWidget(theory_group)
        
        # Circuit parameters
        circuit_group = QtWidgets.QGroupBox("RC Circuit Parameters")
        circuit_layout = QtWidgets.QFormLayout(circuit_group)
        
        self.resistance_spin = QtWidgets.QDoubleSpinBox()
        self.resistance_spin.setRange(100, 100000)
        self.resistance_spin.setValue(1000)
        self.resistance_spin.setSuffix(" Ω")
        self.resistance_spin.setDecimals(0)
        circuit_layout.addRow("Resistance (R):", self.resistance_spin)
        
        self.capacitance_spin = QtWidgets.QDoubleSpinBox()
        self.capacitance_spin.setRange(1e-9, 1e-3)
        self.capacitance_spin.setValue(1e-6)
        self.capacitance_spin.setSuffix(" F")
        self.capacitance_spin.setDecimals(9)
        self.capacitance_spin.setSingleStep(1e-7)
        circuit_layout.addRow("Capacitance (C):", self.capacitance_spin)
        
        # Derived parameters display
        self.tau_label = QtWidgets.QLabel("1.000 ms")
        circuit_layout.addRow("Time Constant (τ):", self.tau_label)
        
        self.fc_label = QtWidgets.QLabel("159.2 Hz")
        circuit_layout.addRow("Cutoff Freq (fc):", self.fc_label)
        
        layout.addWidget(circuit_group)
        
        # Filter type selection
        filter_group = QtWidgets.QGroupBox("Filter Type")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)
        
        self.lowpass_radio = QtWidgets.QRadioButton("Low-pass Filter")
        self.lowpass_radio.setChecked(True)
        self.highpass_radio = QtWidgets.QRadioButton("High-pass Filter")
        
        filter_layout.addWidget(self.lowpass_radio)
        filter_layout.addWidget(self.highpass_radio)
        layout.addWidget(filter_group)
        
        # Signal generator
        signal_group = QtWidgets.QGroupBox("Input Signal")
        signal_layout = QtWidgets.QFormLayout(signal_group)
        
        self.signal_combo = QtWidgets.QComboBox()
        self.signal_combo.addItems(["Sine Wave", "Square Wave", "Step Input", "Impulse"])
        signal_layout.addRow("Signal Type:", self.signal_combo)
        
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(1, 10000)
        self.freq_spin.setValue(100)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.setDecimals(1)
        signal_layout.addRow("Frequency:", self.freq_spin)
        
        self.amplitude_spin = QtWidgets.QDoubleSpinBox()
        self.amplitude_spin.setRange(0.1, 10.0)
        self.amplitude_spin.setValue(1.0)
        self.amplitude_spin.setSuffix(" V")
        self.amplitude_spin.setDecimals(2)
        signal_layout.addRow("Amplitude:", self.amplitude_spin)
        
        layout.addWidget(signal_group)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.start_button = QtWidgets.QPushButton("Start Analysis")
        self.start_button.setCheckable(True)
        self.start_button.setStyleSheet("QPushButton:checked { background-color: #90EE90; }")
        
        self.reset_button = QtWidgets.QPushButton("Reset")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self) -> QtWidgets.QWidget:
        """Create plot area with multiple analysis views"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        # Create plot widget with 3 rows, 2 columns
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # Row 1: Time domain signals (input and output)
        self.input_plot = self.plot_widget.addPlot(row=0, col=0, title="Input Signal x(t)")
        self.input_plot.setLabel('left', 'Amplitude (V)')
        self.input_plot.setLabel('bottom', 'Time (ms)')
        self.input_plot.showGrid(x=True, y=True, alpha=0.3)
        self.input_curve = self.input_plot.plot(pen=pg.mkPen('blue', width=2))
        
        self.output_plot = self.plot_widget.addPlot(row=0, col=1, title="Output Signal y(t)")
        self.output_plot.setLabel('left', 'Amplitude (V)')
        self.output_plot.setLabel('bottom', 'Time (ms)')
        self.output_plot.showGrid(x=True, y=True, alpha=0.3)
        self.output_curve = self.output_plot.plot(pen=pg.mkPen('red', width=2))
        
        # Row 2: Frequency response (magnitude and phase)
        self.magnitude_plot = self.plot_widget.addPlot(row=1, col=0, title="Magnitude Response |H(jω)|")
        self.magnitude_plot.setLabel('left', 'Magnitude (dB)')
        self.magnitude_plot.setLabel('bottom', 'Frequency (Hz)')
        self.magnitude_plot.setLogMode(x=True, y=False)
        self.magnitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.magnitude_curve = self.magnitude_plot.plot(pen=pg.mkPen('green', width=3))
        
        self.phase_plot = self.plot_widget.addPlot(row=1, col=1, title="Phase Response ∠H(jω)")
        self.phase_plot.setLabel('left', 'Phase (degrees)')
        self.phase_plot.setLabel('bottom', 'Frequency (Hz)')
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen('orange', width=3))
        
        # Row 3: Step response and frequency spectrum
        self.step_plot = self.plot_widget.addPlot(row=2, col=0, title="Step Response")
        self.step_plot.setLabel('left', 'Output (V)')
        self.step_plot.setLabel('bottom', 'Time (ms)')
        self.step_plot.showGrid(x=True, y=True, alpha=0.3)
        self.step_curve = self.step_plot.plot(pen=pg.mkPen('purple', width=2))
        
        self.spectrum_plot = self.plot_widget.addPlot(row=2, col=1, title="Input/Output Spectrum")
        self.spectrum_plot.setLabel('left', 'Magnitude (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_plot.setLogMode(x=True, y=False)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.input_spectrum = self.spectrum_plot.plot(pen=pg.mkPen('blue', width=2), name='Input')
        self.output_spectrum = self.spectrum_plot.plot(pen=pg.mkPen('red', width=2), name='Output')
        self.spectrum_plot.addLegend()
        
        return panel
    
    def update_derived_parameters(self, tau: float, fc: float):
        """Update derived parameter displays"""
        self.tau_label.setText(f"{tau*1000:.3f} ms")
        self.fc_label.setText(f"{fc:.1f} Hz")
    
    def update_time_plots(self, time_ms: np.ndarray, input_signal: np.ndarray, output_signal: np.ndarray):
        """Update time domain plots"""
        self.input_curve.setData(time_ms, input_signal)
        self.output_curve.setData(time_ms, output_signal)
    
    def update_frequency_response(self, frequencies: np.ndarray, magnitude_db: np.ndarray, phase_deg: np.ndarray):
        """Update frequency response plots"""
        self.magnitude_curve.setData(frequencies, magnitude_db)
        self.phase_curve.setData(frequencies, phase_deg)
        
        # Only add -3dB line if it doesn't exist
        if not hasattr(self, '_db3_line_added'):
            db3_line = pg.InfiniteLine(pos=-3, angle=0, 
                                      pen=pg.mkPen('red', width=1, style=QtCore.Qt.PenStyle.DashLine))
            self.magnitude_plot.addItem(db3_line)
            self._db3_line_added = True
    
    def update_step_response(self, time_ms: np.ndarray, step_response: np.ndarray):
        """Update step response plot"""
        self.step_curve.setData(time_ms, step_response)
    
    def update_spectrum(self, frequencies: np.ndarray, input_spectrum_db: np.ndarray, output_spectrum_db: np.ndarray):
        """Update spectrum comparison"""
        mask = frequencies > 1  # Avoid DC component
        self.input_spectrum.setData(frequencies[mask], input_spectrum_db[mask])
        self.output_spectrum.setData(frequencies[mask], output_spectrum_db[mask])


# ----------------------
# EDUCATIONAL CONTROLLER
# ----------------------
class EducationalController:
    """Controller for the educational filter application"""
    
    def __init__(self):
        self.view = EducationalFilterView()
        self.filter_model = FirstOrderFilter()
        self.signal_generator = SignalGenerator()
        
        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_analysis)
        
        # Connect signals
        self.connect_signals()
        
        # Initial update
        self.update_filter_parameters()
        self.update_frequency_analysis()
    
    def connect_signals(self):
        """Connect UI signals to handlers"""
        self.view.resistance_spin.valueChanged.connect(self.update_filter_parameters)
        self.view.capacitance_spin.valueChanged.connect(self.update_filter_parameters)
        self.view.lowpass_radio.toggled.connect(self.update_filter_type)
        self.view.highpass_radio.toggled.connect(self.update_filter_type)
        
        self.view.signal_combo.currentTextChanged.connect(self.update_signal_type)
        self.view.freq_spin.valueChanged.connect(self.update_signal_parameters)
        self.view.amplitude_spin.valueChanged.connect(self.update_signal_parameters)
        
        self.view.start_button.toggled.connect(self.toggle_analysis)
        self.view.reset_button.clicked.connect(self.reset_analysis)
    
    def update_filter_parameters(self):
        """Update filter model when circuit parameters change"""
        self.filter_model.R = self.view.resistance_spin.value()
        self.filter_model.C = self.view.capacitance_spin.value()
        self.filter_model.update_parameters()
        
        # Update display
        self.view.update_derived_parameters(self.filter_model.tau, self.filter_model.fc)
        
        # Update frequency analysis
        self.update_frequency_analysis()
        
        # Update status
        self.view.status_bar.showMessage(
            f"τ = {self.filter_model.tau*1000:.3f}ms, fc = {self.filter_model.fc:.1f}Hz"
        )
    
    def update_filter_type(self):
        """Update filter type when radio button changes"""
        if self.view.lowpass_radio.isChecked():
            self.filter_model.filter_type = 'lowpass'
        else:
            self.filter_model.filter_type = 'highpass'
        
        self.update_frequency_analysis()
    
    def update_signal_type(self):
        """Update signal generator type"""
        signal_map = {
            "Sine Wave": "sine",
            "Square Wave": "square", 
            "Step Input": "step",
            "Impulse": "impulse"
        }
        self.signal_generator.signal_type = signal_map[self.view.signal_combo.currentText()]
    
    def update_signal_parameters(self):
        """Update signal generator parameters"""
        self.signal_generator.frequency = self.view.freq_spin.value()
        self.signal_generator.amplitude = self.view.amplitude_spin.value()
    
    def update_frequency_analysis(self):
        """Update frequency response and step response plots"""
        # Frequency response
        frequencies = np.logspace(-1, 4, 1000)  # 0.1 Hz to 10 kHz
        magnitude_db, phase_deg = self.filter_model.frequency_response(frequencies)
        self.view.update_frequency_response(frequencies, magnitude_db, phase_deg)
        
        # Step response
        time_step = np.linspace(-0.001, 0.01, 1000)  # -1ms to 10ms
        step_response = self.filter_model.step_response(time_step)
        time_step_ms = time_step * 1000
        self.view.update_step_response(time_step_ms, step_response)
    
    def update_analysis(self):
        """Real-time analysis update with error handling"""
        try:
            # Generate input signal
            input_signal = self.signal_generator.generate_signal()
            time_ms = self.signal_generator.t * 1000
            
            # Apply filter
            num, den = self.filter_model.transfer_function_coefficients()
            output_signal = signal.lfilter(num, den, input_signal)
            
            # Update time domain plots
            self.view.update_time_plots(time_ms, input_signal, output_signal)
            
            # Update spectrum analysis (less frequently to improve performance)
            if not hasattr(self, '_spectrum_counter'):
                self._spectrum_counter = 0
            
            self._spectrum_counter += 1
            if self._spectrum_counter % 4 == 0:  # Update spectrum every 4th frame (5fps instead of 20fps)
                frequencies = np.fft.rfftfreq(len(input_signal), 1/self.signal_generator.sample_rate)
                input_fft = np.fft.rfft(input_signal)
                output_fft = np.fft.rfft(output_signal)
                
                input_spectrum_db = 20 * np.log10(np.abs(input_fft) + 1e-12)
                output_spectrum_db = 20 * np.log10(np.abs(output_fft) + 1e-12)
                
                self.view.update_spectrum(frequencies, input_spectrum_db, output_spectrum_db)
                
        except Exception as e:
            print(f"Error in update_analysis: {e}")
            self.timer.stop()
            self.view.start_button.setChecked(False)
            self.view.start_button.setText("Start Analysis")
            self.view.status_bar.showMessage(f"Analysis stopped due to error: {str(e)}")
    
    def toggle_analysis(self, checked: bool):
        """Start/stop real-time analysis"""
        if checked:
            try:
                # Reset spectrum counter
                self._spectrum_counter = 0
                
                # Start with slower refresh rate to prevent hanging
                self.timer.start(100)  # 10 fps instead of 20 fps
                self.view.start_button.setText("Stop Analysis")
                self.view.status_bar.showMessage("Real-time analysis running...")
                print("Analysis started successfully")  # Debug output
            except Exception as e:
                print(f"Error starting analysis: {e}")
                self.view.start_button.setChecked(False)
                self.view.status_bar.showMessage(f"Failed to start analysis: {str(e)}")
        else:
            self.timer.stop()
            self.view.start_button.setText("Start Analysis")
            self.view.status_bar.showMessage("Analysis stopped")
            print("Analysis stopped")  # Debug output
    
    def reset_analysis(self):
        """Reset to default parameters"""
        self.timer.stop()
        self.view.start_button.setChecked(False)
        self.view.start_button.setText("Start Analysis")
        
        # Reset to defaults
        self.view.resistance_spin.setValue(1000)
        self.view.capacitance_spin.setValue(1e-6)
        self.view.lowpass_radio.setChecked(True)
        self.view.signal_combo.setCurrentText("Sine Wave")
        self.view.freq_spin.setValue(100)
        self.view.amplitude_spin.setValue(1.0)
        
        self.view.status_bar.showMessage("Reset to default parameters")


# ----------------------
# MAIN APPLICATION
# ----------------------
def main():
    """Main application entry point"""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("First-Order Filter Analysis - ENGR202")
    app.setOrganizationName("Oregon State University - College of Engineering")
    
    # Set application icon and style
    app.setStyle('Fusion')
    
    # Create and show controller
    controller = EducationalController()
    controller.view.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()