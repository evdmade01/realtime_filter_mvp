# First-Order Filter Analysis - ENGR202 Educational Tool

## Overview

This educational application demonstrates first-order RC filter theory and analysis, designed for Oregon State University's ENGR202 course curriculum. It provides interactive visualization of filter behavior in both time and frequency domains.

## Educational Objectives

Students will learn:
- **Transfer Function Analysis**: Understanding H(s) = Y(s)/X(s) for first-order systems
- **Frequency Response**: Visualizing magnitude |H(jω)| and phase ∠H(jω) 
- **Time Domain Behavior**: Step response and transient analysis
- **Circuit Parameters**: Relationship between R, C, τ, and fc
- **Practical Applications**: Real-time signal filtering and analysis

## Theoretical Foundation

### Transfer Functions

**Low-pass Filter:**
```
H(s) = ωc/(s + ωc) = 1/(1 + sRC)
```

**High-pass Filter:**
```
H(s) = s/(s + ωc) = sRC/(1 + sRC)
```

### Key Relationships

- **Time Constant**: τ = RC
- **Cutoff Frequency**: fc = 1/(2πRC)  
- **Angular Frequency**: ωc = 2πfc

### Frequency Response

**Low-pass Magnitude:**
```
|H(jω)| = 1/√(1 + (ω/ωc)²)
```

**High-pass Magnitude:**
```
|H(jω)| = (ω/ωc)/√(1 + (ω/ωc)²)
```

**Phase Response:**
- Low-pass: ∠H(jω) = -arctan(ω/ωc)
- High-pass: ∠H(jω) = 90° - arctan(ω/ωc)

### Step Response

**Low-pass:**
```
y(t) = 1 - e^(-t/τ) for t ≥ 0
```

**High-pass:**
```
y(t) = e^(-t/τ) for t ≥ 0
```

## Application Features

### 1. Interactive Controls

**RC Circuit Parameters:**
- Resistance: 100Ω to 100kΩ
- Capacitance: 1nF to 1mF
- Real-time calculation of τ and fc

**Filter Type Selection:**
- Low-pass filter
- High-pass filter

**Signal Generator:**
- Sine wave, Square wave, Step input, Impulse
- Adjustable frequency and amplitude

### 2. Comprehensive Visualizations

**Six Analysis Plots:**
1. **Input Signal x(t)**: Time domain input visualization
2. **Output Signal y(t)**: Filtered output in time domain
3. **Magnitude Response |H(jω)|**: Frequency response with -3dB reference
4. **Phase Response ∠H(jω)|**: Phase characteristics
5. **Step Response**: Theoretical step response analysis
6. **Input/Output Spectrum**: Frequency domain comparison

### 3. Educational Features

- **Theoretical formulas** displayed in control panel
- **Real-time parameter updates** (τ, fc)
- **-3dB cutoff reference line** on magnitude plots
- **Interactive real-time analysis** with start/stop controls
- **Parameter reset** for classroom demonstrations

## Usage Instructions

### Basic Operation

1. **Launch Application**: Run `RealtimeFilterLecture_ENGR202.exe`
2. **Set Circuit Parameters**: Adjust R and C values
3. **Choose Filter Type**: Select Low-pass or High-pass
4. **Configure Input Signal**: Select signal type and parameters
5. **Start Analysis**: Click "Start Analysis" for real-time operation

### Classroom Demonstrations

**Demonstration 1: Cutoff Frequency Effect**
1. Set R = 1kΩ, C = 1μF (fc = 159.2 Hz)
2. Use sine wave input at various frequencies:
   - 16 Hz (fc/10): Minimal attenuation
   - 159 Hz (fc): -3dB attenuation
   - 1590 Hz (10×fc): Significant attenuation

**Demonstration 2: Time Constant Visualization**
1. Use step input
2. Observe step response: y(t) = 1 - e^(-t/τ)
3. Change R or C to see τ effect on rise time

**Demonstration 3: Low-pass vs High-pass**
1. Use square wave input
2. Compare low-pass (integration) vs high-pass (differentiation)
3. Observe phase relationships

## Technical Specifications

- **Platform**: Windows 10/11
- **File Size**: ~97 MB (optimized executable)
- **Startup Time**: 2-5 seconds
- **Framework**: PyQt6 with pyqtgraph
- **Mathematics**: SciPy signal processing

## Installation

### Option 1: Executable (Recommended for Students)
1. Download `RealtimeFilterLecture_ENGR202.exe`
2. Double-click to run (no installation required)

### Option 2: Python Source (For Instructors)
1. Install requirements: `pip install PyQt6 pyqtgraph numpy scipy`
2. Run: `python realtime_filter_lecture.py`

## Educational Integration

### Course Alignment
- **ENGR202**: Electrical fundamentals, AC circuit analysis
- **Prerequisites**: Basic circuit theory, complex numbers
- **Learning Outcomes**: Filter design, frequency response analysis

### Suggested Exercises

1. **Parameter Sweep**: Calculate fc for different R-C combinations
2. **3dB Point Analysis**: Verify theoretical vs measured cutoff
3. **Phase Analysis**: Understand phase shift implications
4. **Design Challenge**: Design filters for specific applications

### Assessment Questions

1. "If R = 2.2kΩ and C = 470nF, what is the cutoff frequency?"
2. "At what frequency does a low-pass filter exhibit -3dB attenuation?"
3. "How does increasing the time constant affect the step response?"

## Troubleshooting

**Application won't start:**
- Ensure Windows 10/11 compatibility
- Run as administrator if needed

**Plots not updating:**
- Click "Reset" button
- Restart application

**Performance issues:**
- Close other applications
- Reduce update rate in source code if needed

## Author & Support

Developed for Oregon State University College of Engineering ENGR202 curriculum.

**Contact**: Course instructors for technical support and suggestions.

**Version**: 1.0 (Educational Release)
**Last Updated**: September 2025

---

*This educational tool is designed to enhance understanding of first-order filter theory through interactive visualization and real-time analysis.*