import flet as ft
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import signal as scipy_signal
import time

# --- Signal Functions ---
def rect(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)

def tri(t):
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)

def u(t):
    return np.where(t >= 0, 1, 0)

def delta(t):
    # Use a narrow gaussian to approximate delta function for better visualization
    return np.exp(-t**2 / 0.01) / np.sqrt(0.01 * np.pi)

def compute_signal(signal_id, t, custom_params=None):
    if signal_id == "x1":
        return 2 * rect(2*t - 1)
    elif signal_id == "x2":
        return np.sin(np.pi) * rect(t / 2)
    elif signal_id == "x3":
        return tri(2 * t)
    elif signal_id == "x4":
        return u(t - 2)
    elif signal_id == "x5":
        return u(3 - t)
    elif signal_id == "x6":
        return 2*delta(t + 1) - delta(t - 2) + delta(t) - 2*delta(t - 1)
    elif signal_id == "x7":
        return rect((t - 1)/2) - rect((t + 1)/2)
    elif signal_id == "x8":
        return tri(t - 1) - tri(t + 1)
    elif signal_id == "x9":
        return rect(t / 2) - tri(t)
    elif signal_id == "x10":
        return np.exp(-t) * u(t - 2)
    elif signal_id == "x11":
        return np.sin(4 * np.pi * t)  # sin(4πt)
    elif signal_id == "x12":
        return np.cos(2 * np.pi * t) * np.exp(-0.2 * t) * u(t)  # Damped cosine
    elif signal_id == "x13":
        return np.sign(np.sin(2 * np.pi * t))  # Square wave
    elif signal_id == "x14":
        return scipy_signal.sawtooth(2 * np.pi * t, 0.5)  # Triangle wave
    elif signal_id == "x15":
        return np.where(t >= 0, np.exp(-t) * np.sin(2 * np.pi * t), 0)  # Damped sine starting at t=0
    # ENHANCEMENT: Adding more signal types for educational purposes
    elif signal_id == "x16":
        return np.sin(2 * np.pi * t) * np.sin(20 * np.pi * t)  # AM modulation example
    elif signal_id == "x17":
        return np.sinc(t)  # Sinc function
    elif signal_id == "x18":
        return np.exp(-0.1 * t**2) * np.cos(2 * np.pi * t)  # Gaussian windowed cosine
    elif signal_id == "custom":
        # For custom signal with parameters
        if custom_params and "function" in custom_params:
            func_type = custom_params["function"]
            amplitude = float(custom_params.get("amplitude", 1))
            frequency = float(custom_params.get("frequency", 1))
            phase = float(custom_params.get("phase", 0))
            offset = float(custom_params.get("offset", 0))
            
            if func_type == "sin":
                return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
            elif func_type == "cos":
                return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset
            elif func_type == "square":
                return amplitude * scipy_signal.square(2 * np.pi * frequency * t + phase) + offset
            elif func_type == "sawtooth":
                return amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t + phase) + offset
            elif func_type == "gaussian":
                return amplitude * np.exp(-(t**2) / (2 * frequency**2)) + offset
            elif func_type == "exponential":
                return amplitude * np.exp(frequency * t) + offset
            elif func_type == "damped_sin":
                return amplitude * np.exp(-frequency * np.abs(t)) * np.sin(2 * np.pi * 5 * t + phase) + offset
            elif func_type == "chirp":
                return amplitude * np.sin(2 * np.pi * (frequency * t + 0.5 * t**2)) + offset
            # ENHANCEMENT: Adding more custom signal functions
            elif func_type == "sinc":
                return amplitude * np.sinc(frequency * t) + offset
            elif func_type == "impulse_train":
                period = max(0.01, frequency)  # Prevent division by zero
                return amplitude * np.array([1.0 if abs(ti % period) < 0.01 else 0.0 for ti in t]) + offset
    
    return np.zeros_like(t)

# Function to compute signal operations
def compute_operation(op_type, signal1, signal2):
    if op_type == "add":
        return signal1 + signal2
    elif op_type == "subtract":
        return signal1 - signal2
    elif op_type == "multiply":
        return signal1 * signal2
    elif op_type == "convolve":
        return np.convolve(signal1, signal2, mode='same') / (np.sum(signal2) if np.sum(signal2) != 0 else 1)
    # ENHANCEMENT: Adding more operation types
    elif op_type == "correlation":
        return np.correlate(signal1, signal2, mode='same') / (np.sum(signal2**2) if np.sum(signal2**2) != 0 else 1)
    elif op_type == "amplitude_modulation":
        # Amplitude modulation: carrier * (1 + modulation)
        return signal1 * (1 + signal2)
    return signal1

def main(page: ft.Page):
    # Set up the page
    page.title = "Advanced Signal Visualizer"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO  # Make the page scrollable
    
    # State variables
    current_signal = "x1"
    t_range = (-5, 5)  # Default time range
    custom_params = {
        "function": "sin",
        "amplitude": "1",
        "frequency": "1",
        "phase": "0",
        "offset": "0"
    }
    
    # Operation state for signal operations
    operation_enabled = False
    operation_type = "add"
    second_signal = "x3"
    second_signal_params = custom_params.copy()
    
    # Signal information dictionary - equations and descriptions
    signal_info = {
        "x1": {
            "equation": "x₁(t) = 2·Rect(2t - 1)",
            "description": "Rectangular pulse of amplitude 2, shifted by 0.5 and scaled by 2 in time."
        },
        "x2": {
            "equation": "x₂(t) = sin(π)·Rect(t/2)",
            "description": "Rectangular pulse scaled in time by a factor of 1/2, multiplied by sin(π)."
        },
        "x3": {
            "equation": "x₃(t) = Tri(2t)",
            "description": "Triangular pulse scaled in time by a factor of 2."
        },
        "x4": {
            "equation": "x₄(t) = U(t - 2)",
            "description": "Unit step function shifted to the right by 2 units."
        },
        "x5": {
            "equation": "x₅(t) = U(3 - t)",
            "description": "Inverted unit step function with a transition at t=3."
        },
        "x6": {
            "equation": "x₆(t) = 2δ(t+1) - δ(t-2) + δ(t) - 2δ(t-1)",
            "description": "Sum of scaled and shifted delta functions."
        },
        "x7": {
            "equation": "x₇(t) = Rect((t-1)/2) - Rect((t+1)/2)",
            "description": "Difference of two rectangular pulses, creating a bipolar pulse."
        },
        "x8": {
            "equation": "x₈(t) = Tri(t-1) - Tri(t+1)",
            "description": "Difference of two triangular pulses, shifted in opposite directions."
        },
        "x9": {
            "equation": "x₉(t) = Rect(t/2) - Tri(t)",
            "description": "Difference between a rectangular pulse and triangular pulse."
        },
        "x10": {
            "equation": "x₁₀(t) = exp(-t)·U(t-2)",
            "description": "Exponential decay starting at t=2."
        },
        "x11": {
            "equation": "x₁₁(t) = sin(4πt)",
            "description": "Sinusoidal signal with frequency of 2 Hz."
        },
        "x12": {
            "equation": "x₁₂(t) = cos(2πt)·exp(-0.2t)·U(t)",
            "description": "Damped cosine wave starting at t=0."
        },
        "x13": {
            "equation": "x₁₃(t) = sign(sin(2πt))",
            "description": "Square wave with period 1."
        },
        "x14": {
            "equation": "x₁₄(t) = sawtooth(2πt, 0.5)",
            "description": "Triangle wave with period 1."
        },
        "x15": {
            "equation": "x₁₅(t) = exp(-t)·sin(2πt)·U(t)",
            "description": "Damped sine wave starting at t=0."
        },
        # ENHANCEMENT: Adding descriptions for new signals
        "x16": {
            "equation": "x₁₆(t) = sin(2πt)·sin(20πt)",
            "description": "Amplitude modulation: low-frequency sine modulated by high-frequency sine."
        },
        "x17": {
            "equation": "x₁₇(t) = sinc(t)",
            "description": "Sinc function, the Fourier transform of a rectangular pulse."
        },
        "x18": {
            "equation": "x₁₈(t) = exp(-0.1t²)·cos(2πt)",
            "description": "Gaussian windowed cosine wave."
        },
        "custom": {
            "equation": "Custom signal with user-defined parameters",
            "description": "Create your own signal by selecting a function type and parameters."
        },
    }
    
    # Theme toggle function
    def toggle_theme(e):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        theme_icon_button.icon = ft.Icons.DARK_MODE if page.theme_mode == ft.ThemeMode.LIGHT else ft.Icons.LIGHT_MODE
        update_plot()
        page.update()
    
    # Image widget to display the plot
    image_view = ft.Image(
        src="",
        fit=ft.ImageFit.CONTAIN,
    )
    
    # Equation display text
    equation_display = ft.Text(
        signal_info[current_signal]["equation"],
        size=16,
        weight=ft.FontWeight.W_500,
    )
    
    # Description display text
    description_display = ft.Text(
        signal_info[current_signal]["description"],
        size=14,
        italic=True,
    )
    
    # Function to update the equation and description display
    def update_signal_info_display():
        nonlocal current_signal
        equation_display.value = signal_info.get(current_signal, {}).get("equation", "")
        description_display.value = signal_info.get(current_signal, {}).get("description", "")
        page.update()
    
    # Status text for displaying processing information
    status_text = ft.Text("", size=12, color=ft.Colors.GREY)
    
    # Function to compute the signal, plot it, and update the image
    def update_plot():
        nonlocal current_signal, t_range, custom_params, operation_enabled, operation_type, second_signal, second_signal_params
        
        # Update status
        status_text.value = "Processing..."
        page.update()
        
        start_time = time.time()
        
        signal_id = current_signal
        t = np.linspace(t_range[0], t_range[1], 1000)  # Increased resolution
        
        # Compute primary signal
        if signal_id == "custom":
            y1 = compute_signal(signal_id, t, custom_params)
        else:
            y1 = compute_signal(signal_id, t)
        
        # If operations are enabled, compute the second signal and the operation
        if operation_enabled:
            if second_signal == "custom":
                y2 = compute_signal("custom", t, second_signal_params)
            else:
                y2 = compute_signal(second_signal, t)
            
            # Perform the operation
            y = compute_operation(operation_type, y1, y2)
        else:
            y = y1
        
        # Add noise if enabled
        if noise_switch.value:
            noise_level = float(noise_level_slider.value)
            y = y + noise_level * np.random.randn(len(t))
        
        # Apply filtering if enabled
        if filter_switch.value:
            filter_type = filter_type_dropdown.value
            cutoff = float(filter_cutoff_slider.value)
            
            nyquist = 0.5 * (1 / (t[1] - t[0]))
            normalized_cutoff = cutoff / nyquist
            
            # Design and apply the filter
            if filter_type == "lowpass":
                b, a = scipy_signal.butter(4, normalized_cutoff, 'low')
                y = scipy_signal.filtfilt(b, a, y)
            elif filter_type == "highpass":
                b, a = scipy_signal.butter(4, normalized_cutoff, 'high')
                y = scipy_signal.filtfilt(b, a, y)
            elif filter_type == "bandpass":
                b, a = scipy_signal.butter(4, [normalized_cutoff * 0.5, normalized_cutoff], 'band')
                y = scipy_signal.filtfilt(b, a, y)
            # ENHANCEMENT: Added more filter types
            elif filter_type == "notch":
                quality_factor = 30.0  # Q factor
                b, a = scipy_signal.iirnotch(normalized_cutoff, quality_factor)
                y = scipy_signal.filtfilt(b, a, y)
            elif filter_type == "median":
                # Non-linear median filter (good for impulsive noise)
                kernel_size = max(3, int(normalized_cutoff * 20))
                if kernel_size % 2 == 0:  # Ensure odd kernel size
                    kernel_size += 1
                y = scipy_signal.medfilt(y, kernel_size=kernel_size)
        
        # Create a figure with specified background color
        is_dark_mode = page.theme_mode == ft.ThemeMode.DARK
        bg_color = "#1a1a1a" if is_dark_mode else "#ffffff"
        text_color = "#ffffff" if is_dark_mode else "#000000"
        grid_color = "#333333" if is_dark_mode else "#dddddd"
        
        fig = Figure(figsize=(10, 5), facecolor=bg_color)
        canvas = FigureCanvasAgg(fig)
        
        # Create subplots based on view mode
        if view_mode_tabs.selected_index == 0:  # Time domain only
            ax1 = fig.add_subplot(111)
            ax1.set_facecolor(bg_color)
            
            # Plot the signal
            if line_style_dropdown.value == "solid":
                line_style = "-"
            elif line_style_dropdown.value == "dashed":
                line_style = "--"
            elif line_style_dropdown.value == "dotted":
                line_style = ":"
            else:
                line_style = "-"
            
            ax1.plot(t, y, color=line_color_dropdown.value, linewidth=float(line_width_slider.value), linestyle=line_style)
            
            # If operations are enabled, plot the component signals as well
            if operation_enabled and show_components_switch.value:
                ax1.plot(t, y1, color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7, label=f"Signal {current_signal}")
                ax1.plot(t, y2, color="#888888", linewidth=1, linestyle=":", alpha=0.7, label=f"Signal {second_signal}")
                ax1.legend()
            
            # Set axes properties
            ax1.set_xlabel("Time (t)", color=text_color, fontsize=12)
            ax1.set_ylabel("Amplitude", color=text_color, fontsize=12)
            
            # Create title based on whether operations are enabled
            if operation_enabled:
                op_symbols = {"add": "+", "subtract": "-", "multiply": "×", "convolve": "*", 
                            "correlation": "⋆", "amplitude_modulation": "AM"}
                title = f"Signal {current_signal} {op_symbols.get(operation_type, '?')} Signal {second_signal}"
            else:
                title = f"Signal {signal_id}"
            
            ax1.set_title(title, color=text_color, fontsize=14)
            ax1.grid(grid_switch.value, linestyle='--', alpha=0.7, color=grid_color)
            
            if zero_lines_switch.value:
                ax1.axhline(y=0, color=text_color, linestyle='-', alpha=0.3)
                ax1.axvline(x=0, color=text_color, linestyle='-', alpha=0.3)
            
            # Set tick colors
            ax1.tick_params(axis='x', colors=text_color)
            ax1.tick_params(axis='y', colors=text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
                
        elif view_mode_tabs.selected_index == 1:  # Time and frequency domain
            # Time domain plot
            ax1 = fig.add_subplot(211)
            ax1.set_facecolor(bg_color)
            ax1.plot(t, y, color=line_color_dropdown.value, linewidth=float(line_width_slider.value))
            
            if operation_enabled and show_components_switch.value:
                ax1.plot(t, y1, color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7, label=f"Signal {current_signal}")
                ax1.plot(t, y2, color="#888888", linewidth=1, linestyle=":", alpha=0.7, label=f"Signal {second_signal}")
                ax1.legend()
            
            ax1.set_xlabel("Time (t)", color=text_color, fontsize=12)
            ax1.set_ylabel("Amplitude", color=text_color, fontsize=12)
            
            if operation_enabled:
                op_symbols = {"add": "+", "subtract": "-", "multiply": "×", "convolve": "*", 
                            "correlation": "⋆", "amplitude_modulation": "AM"}
                title = f"Time Domain - Signal {current_signal} {op_symbols.get(operation_type, '?')} Signal {second_signal}"
            else:
                title = f"Time Domain - Signal {signal_id}"
                
            ax1.set_title(title, color=text_color, fontsize=14)
            ax1.grid(grid_switch.value, linestyle='--', alpha=0.7, color=grid_color)
            ax1.tick_params(axis='x', colors=text_color)
            ax1.tick_params(axis='y', colors=text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
            
            # Frequency domain plot
            ax2 = fig.add_subplot(212)
            ax2.set_facecolor(bg_color)
            
            # Compute frequency domain
            dt = t[1] - t[0]
            n = len(t)
            Y = np.fft.fftshift(np.fft.fft(y) / n)
            freq = np.fft.fftshift(np.fft.fftfreq(n, dt))
            
            ax2.plot(freq, np.abs(Y), color="#ff5722", linewidth=2)
            
            if operation_enabled and show_components_switch.value:
                Y1 = np.fft.fftshift(np.fft.fft(y1) / n)
                Y2 = np.fft.fftshift(np.fft.fft(y2) / n)
                ax2.plot(freq, np.abs(Y1), color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7, label=f"Signal {current_signal}")
                ax2.plot(freq, np.abs(Y2), color="#888888", linewidth=1, linestyle=":", alpha=0.7, label=f"Signal {second_signal}")
                ax2.legend()
            
            ax2.set_xlabel("Frequency (Hz)", color=text_color, fontsize=12)
            ax2.set_ylabel("Magnitude", color=text_color, fontsize=12)
            ax2.set_title("Frequency Domain", color=text_color, fontsize=14)
            ax2.grid(grid_switch.value, linestyle='--', alpha=0.7, color=grid_color)
            ax2.tick_params(axis='x', colors=text_color)
            ax2.tick_params(axis='y', colors=text_color)
            for spine in ax2.spines.values():
                spine.set_color(text_color)
            
            # Limit the frequency view to relevant frequencies
            freq_limit = float(freq_limit_slider.value)
            ax2.set_xlim(-freq_limit, freq_limit)
        
        elif view_mode_tabs.selected_index == 2:  # 3D view (time-frequency spectrogram)
            ax1 = fig.add_subplot(111)
            ax1.set_facecolor(bg_color)
            
            # Create a spectrogram for time-frequency analysis
            f, t_spec, Sxx = scipy_signal.spectrogram(y, 1/(t[1]-t[0]))
            
            # Plot the spectrogram
            spectrogram = ax1.pcolormesh(t_spec + t_range[0], f, 10 * np.log10(Sxx + 1e-10), 
                                         shading='gouraud', cmap=colormap_dropdown.value)
            cbar = fig.colorbar(spectrogram, ax=ax1)
            cbar.ax.set_ylabel('Power/Frequency (dB/Hz)', color=text_color)
            cbar.ax.tick_params(colors=text_color)
            
            ax1.set_ylabel('Frequency (Hz)', color=text_color)
            ax1.set_xlabel('Time (s)', color=text_color)
            
            if operation_enabled:
                op_symbols = {"add": "+", "subtract": "-", "multiply": "×", "convolve": "*", 
                            "correlation": "⋆", "amplitude_modulation": "AM"}
                title = f'Spectrogram - Signal {current_signal} {op_symbols.get(operation_type, "?")} Signal {second_signal}'
            else:
                title = f'Spectrogram of Signal {signal_id}'
                
            ax1.set_title(title, color=text_color)
            ax1.tick_params(axis='x', colors=text_color)
            ax1.tick_params(axis='y', colors=text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
                
        elif view_mode_tabs.selected_index == 3:  # Phase plot
            ax1 = fig.add_subplot(111)
            ax1.set_facecolor(bg_color)
            
            # Compute phase
            Y = np.fft.fftshift(np.fft.fft(y) / len(y))
            freq = np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))
            phase = np.angle(Y)
            
            # Plot phase
            ax1.plot(freq, phase, color="#9c27b0", linewidth=2)
            
            if operation_enabled and show_components_switch.value:
                Y1 = np.fft.fftshift(np.fft.fft(y1) / len(y1))
                Y2 = np.fft.fftshift(np.fft.fft(y2) / len(y2))
                phase1 = np.angle(Y1)
                phase2 = np.angle(Y2)
                ax1.plot(freq, phase1, color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7, label=f"Signal {current_signal}")
                ax1.plot(freq, phase2, color="#888888", linewidth=1, linestyle=":", alpha=0.7, label=f"Signal {second_signal}")
                ax1.legend()
            
            ax1.set_xlabel("Frequency (Hz)", color=text_color)
            ax1.set_ylabel("Phase (radians)", color=text_color)
            
            if operation_enabled:
                op_symbols = {"add": "+", "subtract": "-", "multiply": "×", "convolve": "*", 
                            "correlation": "⋆", "amplitude_modulation": "AM"}
                title = f'Phase Spectrum - Signal {current_signal} {op_symbols.get(operation_type, "?")} Signal {second_signal}'
            else:
                title = f'Phase Spectrum of Signal {signal_id}'
                
            ax1.set_title(title, color=text_color)
            ax1.grid(grid_switch.value, linestyle='--', alpha=0.7, color=grid_color)
            
            # Limit the frequency view to relevant frequencies
            freq_limit = float(freq_limit_slider.value)
            ax1.set_xlim(-freq_limit, freq_limit)
            ax1.set_ylim(-np.pi, np.pi)
            ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax1.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], color=text_color)
            
            ax1.tick_params(axis='x', colors=text_color)
            ax1.tick_params(axis='y', colors=text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
        
        # ENHANCEMENT: New waterfall plot view
        elif view_mode_tabs.selected_index == 4:  # Waterfall view (3D time-frequency visualization)
            from mpl_toolkits.mplot3d import Axes3D
            
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_facecolor(bg_color)
            
            # Create a spectrogram with multiple time segments
            num_segments = 20
            segment_length = len(t) // num_segments
            
            # Prepare mesh for 3D plot
            T, F = np.meshgrid(
                np.linspace(t_range[0], t_range[1], num_segments),
                np.linspace(0, float(freq_limit_slider.value), 100)
            )
            
            # Calculate spectrograms for each segment
            Z = np.zeros_like(T)
            
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(y))
                segment = y[start_idx:end_idx]
                
                # Compute frequency content
                segment_freq = np.fft.rfftfreq(len(segment), t[1] - t[0])
                segment_fft = np.abs(np.fft.rfft(segment)) / len(segment)
                
                # Interpolate to match our mesh
                max_freq_idx = min(len(segment_freq), np.sum(segment_freq <= float(freq_limit_slider.value)))
                Z[0:max_freq_idx, i] = segment_fft[0:max_freq_idx]
            
            # Plot as surface
            surf = ax1.plot_surface(T, F, Z, cmap=colormap_dropdown.value, 
                                    linewidth=0, antialiased=True, alpha=0.8)
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            cbar.ax.set_ylabel('Magnitude', color=text_color)
            cbar.ax.tick_params(colors=text_color)
            
            # Set labels and title
            ax1.set_xlabel('Time', color=text_color)
            ax1.set_ylabel('Frequency (Hz)', color=text_color)
            ax1.set_zlabel('Magnitude', color=text_color)
            
            if operation_enabled:
                op_symbols = {"add": "+", "subtract": "-", "multiply": "×", "convolve": "*", 
                            "correlation": "⋆", "amplitude_modulation": "AM"}
                title = f'3D Waterfall - Signal {current_signal} {op_symbols.get(operation_type, "?")} Signal {second_signal}'
            else:
                title = f'3D Waterfall Spectrum of Signal {signal_id}'
            
            ax1.set_title(title, color=text_color)
            
            # Set tick colors
            ax1.tick_params(axis='x', colors=text_color)
            ax1.tick_params(axis='y', colors=text_color)
            ax1.tick_params(axis='z', colors=text_color)
            
            # Set the view angle for better visualization
            ax1.view_init(elev=30, azim=-45)
        
       # Adjust layout
        fig.tight_layout()
        
        # Save figure to a buffer, then encode to base64 for display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Update the image source with the base64 data
        image_view.src = f"data:image/png;base64,{img_data}"
        
        # Update status information
        end_time = time.time()
        status_text.value = f"Processing completed in {end_time - start_time:.2f} seconds."
        page.update()
    
    # --- UI Components ---
    
    # Signal selection dropdown
    signal_dropdown = ft.Dropdown(
        label="Select Signal",
        value=current_signal,
        options=[
            ft.dropdown.Option(key, f"Signal {key}") for key in sorted([k for k in signal_info.keys() if k != "custom"])
        ] + [ft.dropdown.Option("custom", "Custom Signal")],
        on_change=lambda e: set_signal(e.control.value),
        width=200,
    )
    
    # Function to set the signal and update the plot
    def set_signal(signal_id):
        nonlocal current_signal
        current_signal = signal_id
        update_signal_info_display()
        update_plot()
    
    # Time range controls
    time_min = ft.TextField(
        label="Min Time",
        value=str(t_range[0]),
        width=100,
        on_submit=lambda e: update_time_range(),
    )
    
    time_max = ft.TextField(
        label="Max Time",
        value=str(t_range[1]),
        width=100,
        on_submit=lambda e: update_time_range(),
    )
    
    # Function to update time range
    def update_time_range():
        nonlocal t_range
        try:
            min_val = float(time_min.value)
            max_val = float(time_max.value)
            if min_val < max_val:
                t_range = (min_val, max_val)
                update_plot()
            else:
                status_text.value = "Error: Min time must be less than max time."
                page.update()
        except ValueError:
            status_text.value = "Error: Time values must be numbers."
            page.update()
    
    # Update time range button
    update_time_button = ft.ElevatedButton(
        "Update Time Range",
        on_click=lambda e: update_time_range(),
    )
    
    # Custom signal parameters
    function_dropdown = ft.Dropdown(
        label="Function Type",
        value=custom_params["function"],
        options=[
            ft.dropdown.Option("sin", "Sine"),
            ft.dropdown.Option("cos", "Cosine"),
            ft.dropdown.Option("square", "Square"),
            ft.dropdown.Option("sawtooth", "Sawtooth"),
            ft.dropdown.Option("gaussian", "Gaussian"),
            ft.dropdown.Option("exponential", "Exponential"),
            ft.dropdown.Option("damped_sin", "Damped Sine"),
            ft.dropdown.Option("chirp", "Chirp"),
            ft.dropdown.Option("sinc", "Sinc"),
            ft.dropdown.Option("impulse_train", "Impulse Train"),
        ],
        width=150,
        on_change=lambda e: update_custom_param("function", e.control.value),
    )
    
    amplitude_field = ft.TextField(
        label="Amplitude",
        value=custom_params["amplitude"],
        width=100,
        on_change=lambda e: update_custom_param("amplitude", e.control.value),
    )
    
    frequency_field = ft.TextField(
        label="Frequency",
        value=custom_params["frequency"],
        width=100,
        on_change=lambda e: update_custom_param("frequency", e.control.value),
    )
    
    phase_field = ft.TextField(
        label="Phase",
        value=custom_params["phase"],
        width=100,
        on_change=lambda e: update_custom_param("phase", e.control.value),
    )
    
    offset_field = ft.TextField(
        label="Offset",
        value=custom_params["offset"],
        width=100,
        on_change=lambda e: update_custom_param("offset", e.control.value),
    )
    
    # Function to update custom parameters
    def update_custom_param(param, value):
        nonlocal custom_params
        custom_params[param] = value
        if current_signal == "custom":
            update_plot()
    
    # Custom signal update button
    update_custom_button = ft.ElevatedButton(
        "Update Custom Signal",
        on_click=lambda e: update_plot(),
    )
    
    # Operation controls
    operation_switch = ft.Switch(
        label="Enable Signal Operations",
        value=operation_enabled,
        on_change=lambda e: toggle_operations(e.control.value),
    )
    
    def toggle_operations(value):
        nonlocal operation_enabled
        operation_enabled = value
        update_plot()
    
    operation_dropdown = ft.Dropdown(
        label="Operation Type",
        value=operation_type,
        options=[
            ft.dropdown.Option("add", "Addition (+)"),
            ft.dropdown.Option("subtract", "Subtraction (-)"),
            ft.dropdown.Option("multiply", "Multiplication (×)"),
            ft.dropdown.Option("convolve", "Convolution (*)"),
            ft.dropdown.Option("correlation", "Correlation (⋆)"),
            ft.dropdown.Option("amplitude_modulation", "Amplitude Modulation (AM)"),
        ],
        width=200,
        on_change=lambda e: set_operation_type(e.control.value),
    )
    
    def set_operation_type(op_type):
        nonlocal operation_type
        operation_type = op_type
        if operation_enabled:
            update_plot()
    
    second_signal_dropdown = ft.Dropdown(
        label="Second Signal",
        value=second_signal,
        options=[
            ft.dropdown.Option(key, f"Signal {key}") for key in sorted([k for k in signal_info.keys() if k != "custom"])
        ] + [ft.dropdown.Option("custom", "Custom Signal")],
        width=200,
        on_change=lambda e: set_second_signal(e.control.value),
    )
    
    def set_second_signal(signal_id):
        nonlocal second_signal
        second_signal = signal_id
        if operation_enabled:
            update_plot()
    
    # Second signal custom parameters
    second_function_dropdown = ft.Dropdown(
        label="Function Type",
        value=second_signal_params["function"],
        options=[
            ft.dropdown.Option("sin", "Sine"),
            ft.dropdown.Option("cos", "Cosine"),
            ft.dropdown.Option("square", "Square"),
            ft.dropdown.Option("sawtooth", "Sawtooth"),
            ft.dropdown.Option("gaussian", "Gaussian"),
            ft.dropdown.Option("exponential", "Exponential"),
            ft.dropdown.Option("damped_sin", "Damped Sine"),
            ft.dropdown.Option("chirp", "Chirp"),
            ft.dropdown.Option("sinc", "Sinc"),
            ft.dropdown.Option("impulse_train", "Impulse Train"),
        ],
        width=150,
        on_change=lambda e: update_second_custom_param("function", e.control.value),
    )
    
    second_amplitude_field = ft.TextField(
        label="Amplitude",
        value=second_signal_params["amplitude"],
        width=100,
        on_change=lambda e: update_second_custom_param("amplitude", e.control.value),
    )
    
    second_frequency_field = ft.TextField(
        label="Frequency",
        value=second_signal_params["frequency"],
        width=100,
        on_change=lambda e: update_second_custom_param("frequency", e.control.value),
    )
    
    second_phase_field = ft.TextField(
        label="Phase",
        value=second_signal_params["phase"],
        width=100,
        on_change=lambda e: update_second_custom_param("phase", e.control.value),
    )
    
    second_offset_field = ft.TextField(
        label="Offset",
        value=second_signal_params["offset"],
        width=100,
        on_change=lambda e: update_second_custom_param("offset", e.control.value),
    )
    
    # Function to update second signal custom parameters
    def update_second_custom_param(param, value):
        nonlocal second_signal_params
        second_signal_params[param] = value
        if operation_enabled and second_signal == "custom":
            update_plot()
    
    # Update second custom signal button
    update_second_custom_button = ft.ElevatedButton(
        "Update Second Custom Signal",
        on_click=lambda e: update_plot(),
    )
    
    # Show component signals switch
    show_components_switch = ft.Switch(
        label="Show Component Signals",
        value=True,
        on_change=lambda e: update_plot(),
    )
    
    # View mode tabs
    view_mode_tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Time Domain"),
            ft.Tab(text="Time & Frequency"),
            ft.Tab(text="Spectrogram"),
            ft.Tab(text="Phase Spectrum"),
            ft.Tab(text="3D Waterfall"),
        ],
        on_change=lambda e: update_plot(),
    )
    
    # Plot styling controls
    grid_switch = ft.Switch(
        label="Show Grid",
        value=True,
        on_change=lambda e: update_plot(),
    )
    
    zero_lines_switch = ft.Switch(
        label="Show Zero Lines",
        value=True,
        on_change=lambda e: update_plot(),
    )
    
    line_color_dropdown = ft.Dropdown(
        label="Line Color",
        value="#2196f3",
        options=[
            ft.dropdown.Option("#2196f3", "Blue"),
            ft.dropdown.Option("#4caf50", "Green"),
            ft.dropdown.Option("#f44336", "Red"),
            ft.dropdown.Option("#9c27b0", "Purple"),
            ft.dropdown.Option("#ff9800", "Orange"),
            ft.dropdown.Option("#607d8b", "Gray"),
        ],
        width=150,
        on_change=lambda e: update_plot(),
    )
    
    line_width_slider = ft.Slider(
        min=1,
        max=5,
        divisions=8,
        value=2,
        label="Line Width: {value}",
        on_change=lambda e: update_plot(),
    )
    
    line_style_dropdown = ft.Dropdown(
        label="Line Style",
        value="solid",
        options=[
            ft.dropdown.Option("solid", "Solid"),
            ft.dropdown.Option("dashed", "Dashed"),
            ft.dropdown.Option("dotted", "Dotted"),
        ],
        width=150,
        on_change=lambda e: update_plot(),
    )
    
    # Colormap for spectrograms
    colormap_dropdown = ft.Dropdown(
        label="Colormap",
        value="viridis",
        options=[
            ft.dropdown.Option("viridis", "Viridis"),
            ft.dropdown.Option("plasma", "Plasma"),
            ft.dropdown.Option("inferno", "Inferno"),
            ft.dropdown.Option("magma", "Magma"),
            ft.dropdown.Option("cividis", "Cividis"),
            ft.dropdown.Option("jet", "Jet"),
            ft.dropdown.Option("hot", "Hot"),
            ft.dropdown.Option("cool", "Cool"),
        ],
        width=150,
        on_change=lambda e: update_plot(),
    )
    
    # Frequency limit slider for frequency domain plots
    freq_limit_slider = ft.Slider(
        min=1,
        max=20,
        divisions=19,
        value=5,
        label="Frequency Limit: {value} Hz",
        on_change=lambda e: update_plot(),
    )
    
    # Noise controls
    noise_switch = ft.Switch(
        label="Add Noise",
        value=False,
        on_change=lambda e: update_plot(),
    )
    
    noise_level_slider = ft.Slider(
        min=0,
        max=1,
        divisions=20,
        value=0.1,
        label="Noise Level: {value}",
        on_change=lambda e: update_plot(),
    )
    
    # Filter controls
    filter_switch = ft.Switch(
        label="Apply Filter",
        value=False,
        on_change=lambda e: update_plot(),
    )
    
    filter_type_dropdown = ft.Dropdown(
        label="Filter Type",
        value="lowpass",
        options=[
            ft.dropdown.Option("lowpass", "Low Pass"),
            ft.dropdown.Option("highpass", "High Pass"),
            ft.dropdown.Option("bandpass", "Band Pass"),
            ft.dropdown.Option("notch", "Notch Filter"),
            ft.dropdown.Option("median", "Median Filter"),
        ],
        width=150,
        on_change=lambda e: update_plot(),
    )
    
    filter_cutoff_slider = ft.Slider(
        min=0.1,
        max=10,
        divisions=99,
        value=2,
        label="Cutoff Frequency: {value} Hz",
        on_change=lambda e: update_plot(),
    )
    
    # Theme toggle button
    theme_icon_button = ft.IconButton(
        icon=ft.Icons.DARK_MODE,
        tooltip="Toggle Theme",
        on_click=toggle_theme,
    )
    
    # Export controls
    export_button = ft.ElevatedButton(
        "Export Plot",
        icon=ft.Icons.DOWNLOAD,
        on_click=lambda e: export_plot(),
    )
    
    def export_plot():
        # Create a higher resolution version of the plot for export
        # Implementation will depend on how you want to handle file saving in your app
        status_text.value = "Exporting plot..."
        page.update()
        
        # For web app, we could trigger a download
        # This is a placeholder - actual implementation would depend on your deployment
        status_text.value = "Export feature not implemented in this version."
        page.update()
    
    # Organize UI layout
    page.add(
        ft.Row([
            ft.Text("Advanced Signal Visualizer", size=24, weight=ft.FontWeight.BOLD),
            theme_icon_button
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        
        # Signal selection and time range controls
        ft.Container(
            content=ft.Column([
                ft.Text("Signal Selection", size=18, weight=ft.FontWeight.W_500),
                ft.Row([
                    signal_dropdown,
                    ft.Container(width=20),
                    time_min,
                    time_max,
                    update_time_button
                ]),
                ft.Row([
                    equation_display
                ]),
                ft.Row([
                    description_display
                ]),
            ]),
            padding=10,
            border=ft.border.all(1, ft.Colors.BLACK12),
            border_radius=ft.border_radius.all(10),
            margin=ft.margin.only(bottom=10),
        ),
        
        # Main visualization area
        ft.Container(
            content=image_view,
            alignment=ft.alignment.center,
            padding=10,
            border=ft.border.all(1, ft.Colors.BLACK12),
            border_radius=ft.border_radius.all(10),
            margin=ft.margin.only(bottom=10),
            width=1000,
            height=500,
        ),
        
        # Plot controls
        ft.Container(
            content=ft.Column([
                ft.Text("Plot Controls", size=18, weight=ft.FontWeight.W_500),
                view_mode_tabs,
                ft.Row([
                    ft.Column([
                        ft.Text("Plot Style", weight=ft.FontWeight.W_500),
                        ft.Row([grid_switch, zero_lines_switch]),
                        ft.Row([
                            line_color_dropdown,
                            line_style_dropdown
                        ]),
                        ft.Row([line_width_slider]),
                    ]),
                    ft.VerticalDivider(width=1, color=ft.Colors.BLACK26),
                    ft.Column([
                        ft.Text("Frequency Domain Options", weight=ft.FontWeight.W_500),
                        ft.Row([freq_limit_slider]),
                        ft.Row([colormap_dropdown]),
                    ]),
                ]),
            ]),
            padding=10,
            border=ft.border.all(1, ft.Colors.BLACK12),
            border_radius=ft.border_radius.all(10),
            margin=ft.margin.only(bottom=10),
        ),
        
        # Advanced options in expandable sections
        ft.ExpansionPanel(
            header=ft.ListTile(title=ft.Text("Custom Signal Parameters")),
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        function_dropdown,
                        amplitude_field,
                        frequency_field,
                        phase_field,
                        offset_field,
                    ]),
                    ft.Row([update_custom_button], alignment=ft.MainAxisAlignment.END),
                ]),
                padding=ft.padding.all(10),
            ),
            expanded=False,
        ),
        
        ft.ExpansionPanel(
            header=ft.ListTile(title=ft.Text("Signal Operations")),
            content=ft.Container(
                content=ft.Column([
                    ft.Row([operation_switch]),
                    ft.Row([
                        operation_dropdown,
                        second_signal_dropdown,
                    ]),
                    ft.Row([show_components_switch]),
                    ft.Text("Second Custom Signal Parameters", weight=ft.FontWeight.W_500),
                    ft.Row([
                        second_function_dropdown,
                        second_amplitude_field,
                        second_frequency_field,
                        second_phase_field,
                        second_offset_field,
                    ]),
                    ft.Row([update_second_custom_button], alignment=ft.MainAxisAlignment.END),
                ]),
                padding=ft.padding.all(10),
            ),
            expanded=False,
        ),
        
        ft.ExpansionPanel(
            header=ft.ListTile(title=ft.Text("Signal Processing")),
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Noise Generation", weight=ft.FontWeight.W_500),
                    ft.Row([
                        noise_switch,
                        noise_level_slider,
                    ]),
                    ft.Text("Filtering", weight=ft.FontWeight.W_500),
                    ft.Row([
                        filter_switch,
                        filter_type_dropdown,
                        filter_cutoff_slider,
                    ]),
                ]),
                padding=ft.padding.all(10),
            ),
            expanded=False,
        ),
        
        # Export and status row
        ft.Row([
            export_button,
            status_text,
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
    )
    
    # Initialize the plot
    update_plot()

if __name__ == "__main__":
    ft.app(target=main)