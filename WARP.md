# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Realtime Filter MVP** - a Python GUI application built with PyQt6 that demonstrates real-time signal processing and digital filter visualization. The application follows an MVP (Model-View-Presenter) architectural pattern and provides an interactive interface for experimenting with different waveforms and digital filters.

## Core Architecture

The codebase is organized around the **MVP (Model-View-Presenter) pattern**:

- **Model** (`Model` class): Handles signal generation, filter design, and DSP operations using NumPy and SciPy
- **View** (`View` class): PyQt6-based GUI with pyqtgraph plotting widgets for real-time visualization
- **Presenter** (`Presenter` class): Coordinates between Model and View, handles user interactions and timer-based updates

### Key Components

- **Signal Generation**: Supports multiple waveform types (sawtooth, triangle, square) with configurable frequency
- **Filter Design**: Implements Butterworth filters (lowpass, highpass, bandpass) using SciPy
- **Real-time Processing**: Uses FFT-based filtering with configurable update rates (~16-17 fps)
- **Visualization**: Three-panel display showing input signal, filter responses, and filtered output in both time and frequency domains

## Development Commands

### Setup and Installation
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```powershell
# Run from project root
python src/realtime_filter_mvp.py

# Or run from src directory
cd src
python realtime_filter_mvp.py
```

### Building Executable
```powershell
# The README mentions build_exe.bat but it's not present in the repository
# To build manually with PyInstaller:
pyinstaller --onefile --windowed --name "RealtimeFilterMVP" src/realtime_filter_mvp.py
```

## Dependencies and Requirements

The project relies on these key libraries:
- **PyQt6**: GUI framework and widgets
- **pyqtgraph**: High-performance real-time plotting
- **NumPy**: Numerical operations and array processing
- **SciPy**: Digital signal processing and filter design
- **PyInstaller**: For creating standalone executables

## Signal Processing Implementation

The application implements frequency-domain filtering using:
- **FFT-based filtering**: `apply_filter_via_fft()` method uses FFT → multiply by filter response → IFFT
- **Butterworth filter design**: Uses `scipy.signal.butter()` with bilinear transform
- **Frequency response computation**: `scipy.signal.freqz()` for filter visualization
- **Real-time buffer processing**: Fixed buffer size of 4096 samples at 44.1kHz sample rate

## GUI Structure

The interface consists of:
- **Control Panel**: Wave type selection, frequency, filter parameters, real-time toggle
- **Three Plot Areas**:
  1. Input signal (time domain)
  2. Filter frequency responses (log-frequency scale)
  3. Output signal (time domain) and spectrum (frequency domain)

## Development Notes

- The application uses a 60ms timer interval for real-time updates
- Filter parameters are normalized to Nyquist frequency for digital filter design
- Frequency plots use log scale to better visualize filter characteristics
- The MVP pattern allows easy extension of signal processing algorithms or GUI components