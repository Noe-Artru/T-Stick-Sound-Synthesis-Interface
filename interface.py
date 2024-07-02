from tkinter import *
from ToggleButton import ToggleButton

# Define a function to handle slider adjustments
def display_value(*args, sliders):
    for i, slider in enumerate(sliders, start=1):
        print(f"Slider {i}: {slider.get()}")

def update_params_from_slider(parameters, slider_vars):
    parameters["f"].value = slider_vars[0].get()
    parameters["c_ratio"].value = slider_vars[1].get()
    parameters["m_ratio"].value = slider_vars[2].get()
    parameters["FMindex"].value = int(slider_vars[3].get())
    parameters["env"].attack = slider_vars[4].get()
    parameters["env"].decay = slider_vars[5].get()
    parameters["env"].sustain = slider_vars[6].get()
    parameters["env"].release = slider_vars[7].get()
    parameters["amp"].value = slider_vars[8].get()
    parameters["env"].dur =  slider_vars[4].get()+slider_vars[5].get()+slider_vars[7].get()
    parameters["env"].play()

def init_slider_window(sliderFrame, buttonFrame, parameters, save_points, load_points, clear_points, toggle_mode, train_decoder, toggle_simulation):
    sliderNames = ["Main Frequency", "Carrier Ratio", "Modulator Ratio", "Modulation Index", "Attack Time", "Decay Time", "Sustain Level", "Release Time", "Volume"]
    sliderRangeMax = [1000, 3, 3, 10, 0.1, 1, 2, 2, 0.4]
    sliderRangeMin = [20, 0.1, 0.1, 0, 0, 0.01, 0, 0.01, 0]
    sliderResolution = [0.1, 0.01, 0.01, 1, 0.001, 0.01, 0.01, 0.01, 0.001]
    sliderDefaultValues = [250, 1, 1, 1, 0.01, 0.1, 0.5, 1.5, 0.2]
    sliders = []
    labels = []
    slider_vars = []

    for i in range(len(sliderNames)):
        slider_var = DoubleVar(value=sliderDefaultValues[i])
        slider = Scale(sliderFrame, from_=sliderRangeMax[i], to=sliderRangeMin[i], orient=VERTICAL, length=150, width=20, resolution=sliderResolution[i], variable=slider_var)
        slider.grid(row=0, column=i, padx=(5, 10))
        sliders.append(slider)
        slider_vars.append(slider_var)
        
        label = Label(sliderFrame, text=sliderNames[i])
        label.grid(row=1, column=i, padx=(10, 0))
        labels.append(label)

    

    run_button = Button(buttonFrame, text="Play Sound", command=lambda: update_params_from_slider(parameters, slider_vars))
    run_button.grid(row=0, column=0, padx=(10, 10), pady=(10, 10))

    save_points_button = Button(buttonFrame, text="Save points", command=save_points)
    save_points_button.grid(row=1, column=0, padx=(10, 10), pady=(10, 10))

    load_points_button = Button(buttonFrame, text="Load points", command=load_points)
    load_points_button.grid(row=2, column=0, padx=(10, 10), pady=(10, 10))

    clear_points_button = Button(buttonFrame, text="Clear points", command=clear_points)
    clear_points_button.grid(row=3, column=0, padx=(10, 10), pady=(10, 10))

    train_button = Button(buttonFrame, text="Train", command=train_decoder)
    train_button.grid(row=4, column=0, padx=(10, 10), pady=(10, 10))

    toggle_button = ToggleButton(buttonFrame, on_text="Interpolation mode", off_text="Training mode", extra_command=toggle_mode)
    toggle_button.button.grid(row=5, column=0, padx=(10, 10), pady=(10, 10))

    toggle_button = ToggleButton(buttonFrame, on_text="Simulation:On", off_text="Simulation:Off", extra_command=toggle_simulation)
    toggle_button.button.grid(row=6, column=0, padx=(10, 10), pady=(10, 10))

    quit_button = Button(buttonFrame, text="Quit", command=quit)
    quit_button.grid(row=7, column=0, padx=(10, 10), pady=(10, 10))

    return sliders


