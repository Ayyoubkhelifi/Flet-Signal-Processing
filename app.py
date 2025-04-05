import flet as ft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import io
import base64
import math
import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Signal processing functions
def sgn(t):
    """Sign function"""
    return np.where(t < 0, -1, np.where(t > 0, 1, 0))

def unit_step(t):
    """Unit step function U(t)"""
    return np.where(t < 0, 0, 1)

def delta(t):
    """Approximation of the delta function"""
    # Using a narrow Gaussian pulse as an approximation
    sigma = 0.05
    return np.where(np.abs(t) < 3*sigma, np.exp(-0.5*(t/sigma)**2)/(sigma*np.sqrt(2*np.pi)), 0)

def rect(t):
    """Rectangle function"""
    return np.where(np.abs(t) <= 0.5, 1, 0)

def tri(t):
    """Triangle function"""
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)

def u_function_using_sgn(t):
    """Express U(t) using sgn(t)"""
    return 0.5 * (1 + sgn(t))

def x1_function_using_u(t):
    """Express x₁(t) = Re c(2t) using U(t)"""
    # Re c(2t) = cos(2πt)
    # This can be expressed in terms of U(t) by using Fourier series approximation
    # For simplicity, we'll just return the direct computation
    return np.cos(2*np.pi*t)

# Signal energy and power calculation
def calculate_energy(func, t_range):
    """Calculate the energy of a signal using the trapezoidal rule"""
    t = np.linspace(t_range[0], t_range[1], 1000)
    y = func(t)
    energy = np.trapz(y**2, t)
    return energy

def calculate_power(func, t_range):
    """Calculate the average power of a signal"""
    t = np.linspace(t_range[0], t_range[1], 1000)
    y = func(t)
    power = np.trapz(y**2, t) / (t_range[1] - t_range[0])
    return power

def classify_signal(func, t_range):
    """Classify a signal as energy signal, power signal, or both"""
    energy = calculate_energy(func, t_range)
    power = calculate_power(func, t_range)
    
    if np.isfinite(energy) and energy > 0 and (power == 0 or not np.isfinite(power)):
        return "Energy Signal", energy, power
    elif energy == float('inf') and np.isfinite(power) and power > 0:
        return "Power Signal", energy, power
    elif np.isfinite(energy) and energy > 0 and np.isfinite(power) and power > 0:
        return "Both Energy and Power Signal", energy, power
    else:
        return "Neither Energy nor Power Signal", energy, power

def generate_plot(func, t_range, title, xlabel="t", ylabel="Amplitude", grid=True, dark_mode=False):
    """Generate a matplotlib plot and convert to base64 for display in Flet"""
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    t = np.linspace(t_range[0], t_range[1], 1000)
    y = func(t)
    
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set dark mode if requested
    if dark_mode:
        fig.patch.set_facecolor('#303030')
        ax.set_facecolor('#303030')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert PNG to base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close(fig)
    return img_str

def main(page: ft.Page):
    page.title = "Signal Processing - Exercise 2"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    page.scroll = "auto"  # Setting page scroll to auto
    
    # Define range for t
    t_min = -5
    t_max = 5
    
    # Function to evaluate x(t), y(t), z(t), w(t)
    def x_function(t):
        return (t**3 - 2*t + 5) * delta(3-t)
    
    def y_function(t):
        return (np.cos(np.pi*t) - t) * delta(t-1)
    
    def z_function(t):
        return (2*t - 1) * delta(t-2)
    
    def w_function(t):
        # For simplicity, we're using rect(t) as the function to convolve with delta
        # In a real application, you might want to implement a proper convolution
        return rect(t) * delta(t-2)
    
    # Signal functions for x₁₁(t) = sin(4πt) from the exercise
    def x11_function(t):
        """x₁₁(t) = sin(4πt) - for determining frequency and period"""
        return np.sin(4*np.pi*t)

    # Define the UI elements we'll need
    dark_mode_switch = ft.Switch(
        label="Dark Mode",
        value=False,
        on_change=lambda e: toggle_theme(e.control.value)
    )
    
    # Create components for tab 1 - Signal Visualization
    function_dropdown = ft.Dropdown(
        width=400,
        label="Select Signal Function",
        options=[
            ft.dropdown.Option("U(t) in terms of sgn(t)"),
            ft.dropdown.Option("x₁(t) in terms of U(t)"),
            ft.dropdown.Option("x(t) = (t³ - 2t + 5)δ(3-t)"),
            ft.dropdown.Option("y(t) = (cos(πt) - t)δ(t-1)"),
            ft.dropdown.Option("z(t) = (2t - 1)δ(t-2)"),
            ft.dropdown.Option("w(t) = rect(t)*δ(t-2)"),
            ft.dropdown.Option("x₁₁(t) = sin(4πt)"),
        ],
        value="U(t) in terms of sgn(t)",
    )
    
    t_min_slider = ft.Slider(
        min=-10.0,
        max=0.0,
        value=t_min,
        divisions=20,
        label="{value}",
        width=400,
    )
    
    t_max_slider = ft.Slider(
        min=0.0,
        max=10.0,
        value=t_max,
        divisions=20,
        label="{value}",
        width=400,
    )
    
    range_text = ft.Text(f"t range: [{t_min}, {t_max}]")
    
    formula_text = ft.Text(
        "U(t) = 0.5 * (1 + sgn(t))",
        size=18,
        weight=ft.FontWeight.BOLD,
    )
    
    plot_image = ft.Image(
        src_base64=generate_plot(
            u_function_using_sgn,
            [t_min, t_max],
            "U(t) in terms of sgn(t)",
            dark_mode=False
        ),
        width=800,
        height=400,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # Create components for tab 2 - Energy & Power
    energy_dropdown = ft.Dropdown(
        width=400,
        label="Select Signal Function for Energy Analysis",
        options=[
            ft.dropdown.Option("U(t) in terms of sgn(t)"),
            ft.dropdown.Option("x₁(t) in terms of U(t)"),
            ft.dropdown.Option("x(t) = (t³ - 2t + 5)δ(3-t)"),
            ft.dropdown.Option("y(t) = (cos(πt) - t)δ(t-1)"),
            ft.dropdown.Option("z(t) = (2t - 1)δ(t-2)"),
            ft.dropdown.Option("w(t) = rect(t)*δ(t-2)"),
            ft.dropdown.Option("x₁₁(t) = sin(4πt)"),
        ],
        value="U(t) in terms of sgn(t)",
    )
    
    classification_text = ft.Text(
        "Click 'Calculate' to analyze signal energy & power",
        size=16,
    )
    
    energy_text = ft.Text(
        "Energy: -",
        size=16,
    )
    
    power_text = ft.Text(
        "Power: -",
        size=16,
    )
    
    calculate_button = ft.ElevatedButton(
        "Calculate Energy & Power",
        icon=ft.Icons.CALCULATE_OUTLINED,
    )
    
    energy_plot_image = ft.Image(
        width=800,
        height=400,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # Create components for tab 3 - Frequency Analysis
    frequency_text = ft.Text(
        "Frequency (f₀): 2 Hz",
        size=16,
    )
    
    period_text = ft.Text(
        "Period (T): 0.5 seconds",
        size=16,
    )
    
    frequency_plot_image = ft.Image(
        src_base64=generate_plot(
            x11_function,
            [-1, 1],
            "x₁₁(t) = sin(4πt) - Frequency: 2 Hz, Period: 0.5s",
            dark_mode=False
        ),
        width=800,
        height=400,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # AI Assistant Components
    ai_signal_dropdown = ft.Dropdown(
        width=400,
        label="Select Signal for AI Explanation",
        options=[
            ft.dropdown.Option("U(t) in terms of sgn(t)"),
            ft.dropdown.Option("x₁(t) in terms of U(t)"),
            ft.dropdown.Option("x(t) = (t³ - 2t + 5)δ(3-t)"),
            ft.dropdown.Option("y(t) = (cos(πt) - t)δ(t-1)"),
            ft.dropdown.Option("z(t) = (2t - 1)δ(t-2)"),
            ft.dropdown.Option("w(t) = rect(t)*δ(t-2)"),
            ft.dropdown.Option("x₁₁(t) = sin(4πt)"),
        ],
        value="U(t) in terms of sgn(t)",
    )
    
    ask_button = ft.ElevatedButton(
        "Generate Explanation",
        icon=ft.Icons.SMART_TOY_OUTLINED,
    )
    
    ai_response_text = ft.Markdown(
        "Select a signal function and click 'Generate Explanation' to learn more...",
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme="atom-one-dark" if page.theme_mode == ft.ThemeMode.DARK else "github",
    )
    
    ai_thinking_progress = ft.ProgressRing(width=24, height=24, visible=False)
    
    ai_explanation_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ai_response_text,
            ]),
            padding=20,
            expand=True,
            width=800,
        ),
        expand=True,
    )
    
    # Status text for debugging
    ai_status_text = ft.Text(
        "",
        size=12,
        color=ft.Colors.GREY_500,
        visible=False,
    )
    
    # OpenRouter API integration
    OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found in environment variables. AI features will not work.")
    
    def get_signal_details(signal_name):
        """Get specific details about each signal function for the AI prompt"""
        details = {
            "U(t) in terms of sgn(t)": {
                "definition": "U(t) = 0.5 * (1 + sgn(t))",
                "implementation": "np.where(t < 0, 0, 1)",
                "function": "Unit step function expressed using sign function"
            },
            "x₁(t) in terms of U(t)": {
                "definition": "x₁(t) = cos(2πt) approximated using U(t)",
                "implementation": "np.cos(2*np.pi*t)",
                "function": "Cosine function expressed in terms of unit step"
            },
            "x(t) = (t³ - 2t + 5)δ(3-t)": {
                "definition": "Polynomial multiplied by shifted delta function",
                "implementation": "(t**3 - 2*t + 5) * delta(3-t)",
                "function": "Product of polynomial and shifted delta function"
            },
            "y(t) = (cos(πt) - t)δ(t-1)": {
                "definition": "Cosine minus t, multiplied by shifted delta",
                "implementation": "(np.cos(np.pi*t) - t) * delta(t-1)",
                "function": "Product of cosine minus t and shifted delta function"
            },
            "z(t) = (2t - 1)δ(t-2)": {
                "definition": "Linear function multiplied by shifted delta",
                "implementation": "(2*t - 1) * delta(t-2)",
                "function": "Product of linear function and shifted delta"
            },
            "w(t) = rect(t)*δ(t-2)": {
                "definition": "Rectangle function multiplied by shifted delta",
                "implementation": "rect(t) * delta(t-2)",
                "function": "Product of rectangle function and shifted delta"
            },
            "x₁₁(t) = sin(4πt)": {
                "definition": "Sinusoidal function with frequency 2Hz and period 0.5s",
                "implementation": "np.sin(4*np.pi*t)",
                "function": "Sine function with frequency 2Hz"
            },
        }
        return details.get(signal_name, {"definition": "Unknown function", "implementation": "Unknown", "function": "Unknown"})
    
    def stream_ai_response(signal_name):
        """Generate and stream AI response for explaining the selected signal"""
        signal_details = get_signal_details(signal_name)
        
        # Show the thinking indicator
        ai_thinking_progress.visible = True
        ai_response_text.value = "Generating explanation..."
        ai_status_text.value = "Sending request..."
        ai_status_text.visible = True
        page.update()
        
        # Construct the prompt for the AI
        prompt = f"""
        Explain the signal {signal_name} in signal processing:
        
        Definition: {signal_details['definition']}
        Implementation: {signal_details['implementation']}
        Function Type: {signal_details['function']}
        
        Please cover:
        1. What does this signal represent in signal processing?
        2. What are its key features and properties?
        3. How is it calculated or derived mathematically?
        4. What are common practical applications?
        5. How does it relate to other signal functions?
        
        Format your response with markdown headings.
        
        IMPORTANT FOR MATHEMATICAL EXPRESSIONS:
        - Do NOT use LaTeX format
        - Use plain text for all mathematics
        - For example, write x^2 instead of x²
        - For fractions, use a/b instead of ⅘ or LaTeX fractions
        - For equations, use plain text like: y(t) = 3*t^2 + 2*t + 1
        - For special symbols, spell them out (e.g., "pi" instead of π)
        - When you need to highlight a mathematical expression, use code blocks with the language set to "text":
        
        ```text
        y(t) = sin(2*pi*t)
        ```
        
        Provide a comprehensive explanation for students learning signal processing.
        """
        
        # Prepare the request to OpenRouter API
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'HTTP-Referer': 'https://signal.app',
            'X-Title': 'signal App',
            'Accept': 'text/event-stream',  # Important for SSE
        }
        
        data = {
            "model": "deepseek/deepseek-chat",  # Using a more reliable model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800,
            "stream": True  # Request streaming response
        }
        
        try:
            # Clear previous response
            ai_response_text.value = ""
            page.update()
            
            # Send request to OpenRouter API
            ai_status_text.value = "Connecting to AI service..."
            page.update()
            
            response = requests.post(
                f"{OPENROUTER_API_URL}/chat/completions",
                headers=headers,
                json=data,
                stream=True  # Enable streaming
            )
            
            if response.status_code != 200:
                ai_response_text.value = f"Error: API returned status code {response.status_code}\n{response.text}"
                ai_status_text.value = f"Error: Status code {response.status_code}"
                page.update()
                ai_thinking_progress.visible = False
                return
                
            ai_status_text.value = "Receiving response..."
            page.update()
                
            # Process the streaming response
            buffer = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # Handle OpenRouter processing messages
                    if ": OPENROUTER PROCESSING" in line_text:
                        ai_status_text.value = "Processing request..."
                        continue
                    
                    # Handle data chunks
                    if line_text.startswith('data: '):
                        line_json = line_text[6:]  # Remove 'data: ' prefix
                        
                        # Check for [DONE] message
                        if line_json.strip() == '[DONE]':
                            break
                        
                        try:
                            # Parse JSON response
                            json_response = json.loads(line_json)
                            if 'choices' in json_response and len(json_response['choices']) > 0:
                                delta = json_response['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    buffer += content
                                    
                                    # Update the UI with new content
                                    ai_response_text.value = buffer
                                    page.update()
                        except json.JSONDecodeError:
                            continue
                    elif "OPENROUTER" not in line_text:  # Ignore OPENROUTER messages
                        # Debug other unexpected lines
                        print(f"Unexpected line format: {line_text}")
            
            # Post-process the response to enhance math rendering
            final_response = buffer
            
            # Convert math code blocks to prettier format
            final_response = enhance_math_rendering(final_response)
            
            # Update with enhanced response
            ai_response_text.value = final_response
            ai_status_text.value = "Completed successfully"
            page.update()
            
        except Exception as e:
            ai_response_text.value = f"Error generating explanation: {str(e)}"
            ai_status_text.value = f"Exception: {str(e)}"
            print(f"Exception in AI streaming: {str(e)}")
            page.update()
        finally:
            # Hide the thinking indicator
            ai_thinking_progress.visible = False
            page.update()
            
            # Hide status after 3 seconds
            time.sleep(3)
            ai_status_text.visible = False
            page.update()
            
    def enhance_math_rendering(text):
        """Enhance the rendering of math expressions in the response"""
        # Make equations stand out more
        lines = text.split('\n')
        in_math_block = False
        result = []
        
        for line in lines:
            if line.strip() == '```text' or line.strip() == '```math':
                in_math_block = True
                result.append('**Equation:**')
                result.append('```')
                continue
            elif line.strip() == '```' and in_math_block:
                in_math_block = False
                result.append('```')
                continue
            
            if in_math_block:
                # Clean up the equation line and make it more readable
                eq_line = line.strip()
                # Replace common math symbols with more readable alternatives if needed
                eq_line = eq_line.replace('*', '·').replace('^', '^')
                result.append(eq_line)
            else:
                # Look for inline math patterns and make them more readable
                if '=' in line and ('(' in line or ')' in line or '+' in line or '-' in line):
                    parts = []
                    for part in line.split():
                        if '=' in part and any(c in part for c in '()+-*/^'):
                            # This part looks like a mathematical expression
                            parts.append(f"`{part}`")
                        else:
                            parts.append(part)
                    line = ' '.join(parts)
                result.append(line)
                
        return '\n'.join(result)
    
    # Connect the button to the AI function
    ask_button.on_click = lambda e: stream_ai_response(ai_signal_dropdown.value)
    
    # Define the update functions
    def update_t_range(e):
        nonlocal t_min, t_max
        t_min = t_min_slider.value
        t_max = t_max_slider.value
        range_text.value = f"t range: [{t_min:.1f}, {t_max:.1f}]"
        update_plot()
        page.update()
    
    # Connect sliders to update function
    t_min_slider.on_change = update_t_range
    t_max_slider.on_change = update_t_range
    
    def update_plot():
        selected_function = function_dropdown.value
        
        if selected_function == "U(t) in terms of sgn(t)":
            func = u_function_using_sgn
            title = "U(t) in terms of sgn(t)"
            formula_text.value = "U(t) = 0.5 * (1 + sgn(t))"
        elif selected_function == "x₁(t) in terms of U(t)":
            func = x1_function_using_u
            title = "x₁(t) = Re c(2t) in terms of U(t)"
            formula_text.value = "x₁(t) = cos(2πt) approximated using U(t)"
        elif selected_function == "x(t) = (t³ - 2t + 5)δ(3-t)":
            func = x_function
            title = "x(t) = (t³ - 2t + 5)δ(3-t)"
            formula_text.value = "x(t) = (t³ - 2t + 5)δ(3-t)"
        elif selected_function == "y(t) = (cos(πt) - t)δ(t-1)":
            func = y_function
            title = "y(t) = (cos(πt) - t)δ(t-1)"
            formula_text.value = "y(t) = (cos(πt) - t)δ(t-1)"
        elif selected_function == "z(t) = (2t - 1)δ(t-2)":
            func = z_function
            title = "z(t) = (2t - 1)δ(t-2)"
            formula_text.value = "z(t) = (2t - 1)δ(t-2)"
        elif selected_function == "x₁₁(t) = sin(4πt)":
            func = x11_function
            title = "x₁₁(t) = sin(4πt)"
            formula_text.value = "x₁₁(t) = sin(4πt) - Frequency: 2 Hz, Period: 0.5s"
        else:  # w(t) = rect(t)*δ(t-2)
            func = w_function
            title = "w(t) = rect(t)*δ(t-2)"
            formula_text.value = "w(t) = rect(t)*δ(t-2) (convolution)"
        
        is_dark = page.theme_mode == ft.ThemeMode.DARK
        plot_image.src_base64 = generate_plot(
            func, 
            [t_min, t_max], 
            title, 
            dark_mode=is_dark
        )
        page.update()
    
    # Connect dropdown to update function
    function_dropdown.on_change = lambda e: update_plot()
    
    def calculate_energy_power(e):
        selected_function = energy_dropdown.value
        
        if selected_function == "U(t) in terms of sgn(t)":
            func = u_function_using_sgn
            title = "Energy Analysis: U(t) in terms of sgn(t)"
        elif selected_function == "x₁(t) in terms of U(t)":
            func = x1_function_using_u
            title = "Energy Analysis: x₁(t) = Re c(2t)"
        elif selected_function == "x(t) = (t³ - 2t + 5)δ(3-t)":
            func = x_function
            title = "Energy Analysis: x(t) = (t³ - 2t + 5)δ(3-t)"
        elif selected_function == "y(t) = (cos(πt) - t)δ(t-1)":
            func = y_function
            title = "Energy Analysis: y(t) = (cos(πt) - t)δ(t-1)"
        elif selected_function == "z(t) = (2t - 1)δ(t-2)":
            func = z_function
            title = "Energy Analysis: z(t) = (2t - 1)δ(t-2)"
        elif selected_function == "x₁₁(t) = sin(4πt)":
            func = x11_function
            title = "Energy Analysis: x₁₁(t) = sin(4πt)"
        else:  # w(t) = rect(t)*δ(t-2)
            func = w_function
            title = "Energy Analysis: w(t) = rect(t)*δ(t-2)"
        
        classification, energy, power = classify_signal(func, [t_min, t_max])
        
        classification_text.value = f"Classification: {classification}"
        energy_text.value = f"Energy: {energy:.6f}" if np.isfinite(energy) else "Energy: ∞"
        power_text.value = f"Power: {power:.6f}" if np.isfinite(power) else "Power: ∞"
        
        # Generate energy visualization
        is_dark = page.theme_mode == ft.ThemeMode.DARK
        
        # Create a plot showing the signal and its squared value (energy density)
        fig = Figure(figsize=(10, 6), dpi=100)
        t = np.linspace(t_min, t_max, 1000)
        y = func(t)
        
        ax1 = fig.add_subplot(211)
        ax1.plot(t, y, 'b-')
        ax1.set_title(f"Signal: {selected_function}")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = fig.add_subplot(212)
        ax2.plot(t, y**2, 'r-')
        ax2.set_title("Energy Density: |x(t)|²")
        ax2.set_xlabel("t")
        ax2.set_ylabel("Energy Density")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Set dark mode if requested
        if is_dark:
            fig.patch.set_facecolor('#303030')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#303030')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                ax.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Convert PNG to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        plt.close(fig)
        energy_plot_image.src_base64 = img_str
        
        page.update()
    
    # Connect button to calculate function
    calculate_button.on_click = calculate_energy_power
    
    def toggle_theme(is_dark):
        page.theme_mode = ft.ThemeMode.DARK if is_dark else ft.ThemeMode.LIGHT
        update_plot()  # Regenerate plot with appropriate theme
        
        # Also update energy plot if it exists
        if energy_plot_image.src_base64:
            calculate_energy_power(None)
            
        # Update frequency plot
        is_dark = page.theme_mode == ft.ThemeMode.DARK
        frequency_plot_image.src_base64 = generate_plot(
            x11_function,
            [-1, 1],
            "x₁₁(t) = sin(4πt) - Frequency: 2 Hz, Period: 0.5s",
            dark_mode=is_dark
        )
        
        # Update markdown code theme
        ai_response_text.code_theme = "atom-one-dark" if is_dark else "github"
        
        page.update()
    
    # Create tabs with the components
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Signal Visualization",
                icon=ft.Icons.ANALYTICS_OUTLINED,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Signal Selection", size=20, weight=ft.FontWeight.BOLD),
                        function_dropdown,
                        formula_text,
                        
                        ft.Container(height=20),
                        
                        ft.Text("Time Range", size=20, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            ft.Text("Minimum t:"),
                            t_min_slider,
                        ]),
                        ft.Row([
                            ft.Text("Maximum t:"),
                            t_max_slider,
                        ]),
                        range_text,
                        
                        ft.Container(height=20),
                        
                        ft.Text("Signal Visualization", size=20, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=plot_image,
                            alignment=ft.alignment.center,
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=10,
                            padding=10,
                        ),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=10,
                    ),
                    padding=20,
                    height=page.height,
                    expand=True,
                )
            ),
            ft.Tab(
                text="Energy & Power",
                icon=ft.Icons.BOLT_OUTLINED,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Signal Energy & Power Analysis", size=20, weight=ft.FontWeight.BOLD),
                        energy_dropdown,
                        
                        ft.Container(height=20),
                        
                        ft.Text("Energy Classification", size=20, weight=ft.FontWeight.BOLD),
                        classification_text,
                        energy_text,
                        power_text,
                        
                        ft.Container(height=20),
                        
                        calculate_button,
                        
                        ft.Container(height=20),
                        
                        ft.Text("Energy Visualization", size=20, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=energy_plot_image,
                            alignment=ft.alignment.center,
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=10,
                            padding=10,
                        ),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=10,
                    ),
                    padding=20,
                    height=page.height,
                    expand=True,
                )
            ),
            ft.Tab(
                text="Frequency Analysis",
                icon=ft.Icons.WAVES_OUTLINED,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Frequency Analysis for x₁₁(t) = sin(4πt)", size=20, weight=ft.FontWeight.BOLD),
                        
                        ft.Container(height=20),
                        
                        ft.Text("Signal Parameters:", size=18, weight=ft.FontWeight.BOLD),
                        frequency_text,
                        period_text,
                        
                        ft.Container(height=20),
                        
                        ft.Container(
                            content=frequency_plot_image,
                            alignment=ft.alignment.center,
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=10,
                            padding=10,
                        ),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=10,
                    ),
                    padding=20,
                    height=page.height,
                    expand=True,
                )
            ),
            ft.Tab(
                text="AI Assistant",
                icon=ft.Icons.SMART_TOY,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("AI Signal Explanation", size=20, weight=ft.FontWeight.BOLD),
                        
                        ft.Container(height=20),
                        
                        ft.Row([
                            ai_signal_dropdown,
                            ft.Container(width=10),
                            ask_button,
                            ft.Container(width=10),
                            ai_thinking_progress,
                        ]),
                        
                        ai_status_text,
                        
                        ft.Container(height=20),
                        
                        ft.Text("AI Explanation", size=20, weight=ft.FontWeight.BOLD),
                        ai_explanation_card,
                        
                        ft.Container(height=40),  # Add extra space at bottom
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=10,
                    ),
                    padding=20,
                    height=page.height,
                    expand=True,
                )
            ),
        ],
        expand=1,
    )
    
    # Create layout with scrolling
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Text("Signal Processing - Exercise 2", size=30, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        dark_mode_switch,
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=20,
                ),
                ft.Divider(),
                tabs,
            ],
            spacing=10,
            ),
            expand=True,
            height=page.height,
        )
    )
    
    # Register to handle page resize events
    def page_resize(e):
        for tab in tabs.tabs:
            if isinstance(tab.content, ft.Container):
                tab.content.height = page.height
        page.update()
        
    page.on_resize = page_resize
    page.update()

ft.app(target=main, view=ft.AppView.WEB_BROWSER)
