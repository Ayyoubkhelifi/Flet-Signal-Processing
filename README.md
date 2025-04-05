# Signal Processing Visualization App

An interactive web application for visualizing signal processing functions from Exercise 2. Built with Flet and Python.

## Features

- **Signal Visualization Tab**:

  - Visualize various signal functions including:
    - U(t) expressed in terms of sgn(t)
    - x₁(t) expressed in terms of U(t)
    - x(t) = (t³ - 2t + 5)δ(3-t)
    - y(t) = (cos(πt) - t)δ(t-1)
    - z(t) = (2t - 1)δ(t-2)
    - w(t) = rect(t)\*δ(t-2)
    - x₁₁(t) = sin(4πt)
  - Adjustable time range for detailed signal analysis

- **Energy & Power Tab**:

  - Analyze energy and power properties of signals
  - Classify signals as energy signals, power signals, both, or neither
  - Visualize signal energy density

- **Frequency Analysis Tab**:

  - Analyze frequency and period of x₁₁(t) = sin(4πt)
  - Visual representation of periodic signals

- **UI Features**:
  - Dark/Light mode toggle for better viewing experience
  - Tabbed interface for organized functionality
  - Interactive controls and real-time updates
  - Responsive design

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Usage

- **Signal Visualization**:

  - Select a signal function from the dropdown menu
  - Adjust the time range using the sliders
  - View the selected signal's mathematical formula and visualization

- **Energy & Power Analysis**:

  - Select a signal for energy analysis
  - Click the "Calculate Energy & Power" button
  - View the signal classification, energy, and power values
  - Examine the energy density visualization

- **Frequency Analysis**:
  - View the frequency and period of the sin(4πt) signal
  - Examine the signal visualization

## Technologies Used

- Flet: For building the interactive web UI
- NumPy: For numerical computations
- Matplotlib: For plotting signal visualizations
- SciPy: For integration and signal processing utilities
