from pyo import *
from tkinter import *
# Import interface code from interface.py
from interface import *
from PIL import ImageTk, Image
import os
from DecoderModel import DecoderNN
import torch
import time

# Connection setup
s = Server(sr=48000, buffersize=1024, duplex=1, winhost="asio").boot()
port = 8000
tStickId = "/TStick_195"

# Variable initialization for parameters, synthesizer, inputs, points and models

parameters = {
    "f": Sig(250),
    "c_ratio": Sig(1),
    "m_ratio": Sig(1),
    "amp": Sig(0.2),
    "env": Adsr(attack=0.01, decay=0.1, sustain=0.5, release=1.5, dur=2, mul=0.5),
    "FMindex": Sig(1),  
}

parameters["env"].setExp(0.75)


sig = FM(carrier=[parameters["c_ratio"]*parameters["f"],parameters["c_ratio"]*parameters["f"]], 
         ratio=[parameters["m_ratio"]/parameters["c_ratio"], parameters["m_ratio"]/parameters["c_ratio"]], 
         index=parameters["FMindex"], mul=parameters["amp"]*parameters["env"]).out()

input = {
    "raw": {
        "fsr": 0,
        "accl": [],
        "gyro": [],
        "magn": [],
        "capsense": [],
    },
    "instrument": {
        "squeeze": 0,
        "touch": {
            "all": 0,
            "top": 0,
            "middle": 0,
            "bottom": 0,
            "normalised": [],
            "discrete": [],
        },
        "shakexyz": [],
        "jabxyz": [],
        "button": {
            "count": [],
            "tap": [],
            "dtap": [],
            "ttap": []
        },
        "brush": 0,
        "multibrush": [],
        "rub": 0,
        "multirub": [],
    },
    "orientation": [],
    "ypr": [],
    #"battery": 0,
    "battery": {
        "percentage": 0,
        "voltage": 0,
    },
}

points = []
point_params = []

is_predicting = False
is_simulating = False

decoderModel = DecoderNN()
current_state = {
    "x": 0.5,
    "y": 0.5,
    "v_x": 0.0005,
    "v_y": 0.0007,
}

optimizer = torch.optim.Adam(decoderModel.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

def has_new_note(input_dict, path, values):
    return input_dict[path[0]].count(0) > values.count(0)

# Main function to handle how the tStick interacts with the interface
def tStickInteraction(input_dict, path, values):
    if path[0] == "capsense" and has_new_note(input_dict, path, values):
        play_note()
    elif path[0] == "fsr":
        current_state["v_x"] = current_state["v_x"]*(1-values*0.00001)
        current_state["v_y"] = current_state["v_y"]*(1-values*0.00001)
    elif path[0] == "jabxyz":
        current_state["v_x"] = current_state["v_x"]+values[0]*0.0002
        current_state["v_y"] = current_state["v_y"]+values[1]*0.0002
    elif path[0] == "shakexyz":
        current_state["v_x"] = current_state["v_x"]*(1-values[0]*0.00003) + current_state["v_y"]*values[0]*0.00003
        current_state["v_y"] = current_state["v_y"]*(1-values[1]*0.00003) + current_state["v_x"]*values[1]*0.00003
    elif path[0] == "brush":
        target_point = [0.59, 0.13]
        current_state["v_x"] = current_state["v_x"] + (target_point[0]-current_state["x"])*0.0001
        current_state["v_y"] = current_state["v_y"] + (target_point[1]-current_state["y"])*0.0001

# Update input dictionary using OSC data
def fill_dict(input_dict, path, values):
    if len(path) == 1:
        tStickInteraction(input_dict, path, values)
        try: input_dict[path[0]] = values
        except Exception as e:
            print("Error:", e)
            print("Path:", path)
    else:
        fill_dict(input_dict[path[0]], path[1:], values)
        
def updateInput(address, *args):
    address = address.split("/")[2:]
    if len(args) == 1:
        args = args[0]
    else:
        args = list(args)

    fill_dict(input, address, args)

def play_note():
    parameters["env"].play() # Plays the note

s.start()
window = Tk()
window.title("T-Stick Sound Synthesis Interface")

sliderFrame = Frame(window)
buttonFrame = Frame(window)
mapDim = {"width": 550, "height": 500}
map2DFrame = Frame(window, width=mapDim["width"], height=mapDim["height"], highlightbackground="gray", highlightthickness=2, cursor="crosshair")


canvas = Canvas(map2DFrame, width=mapDim["width"], height=mapDim["height"])
canvas.pack()


ball_size = 7

def update_params_from_point(point_params):
    for j in range(len(sliders)):
        sliders[j].set(point_params[j])

    parameters["f"].value = point_params[0]
    parameters["c_ratio"].value = point_params[1]
    parameters["m_ratio"].value = point_params[2]
    parameters["FMindex"].value = point_params[3]
    parameters["env"].attack = point_params[4]
    parameters["env"].decay = point_params[5]
    parameters["env"].sustain = point_params[6]
    parameters["env"].release = point_params[7]
    parameters["amp"].value = point_params[8]
    parameters["env"].dur =  point_params[4]+point_params[5]+point_params[7]

def cursor_click(event): 
    # On cursor click, draws a red circle and saves the point. If you click on a point, it plays back its associated sound 
    if(is_predicting):
        interpolate(event)
        return

    for i in range(len(points)):
        if (event.x-points[i][0]*mapDim["width"])**2 + (event.y-points[i][1]*mapDim["height"])**2 < ball_size**2:
            update_params_from_point(point_params[i])
            play_note()
            return
    
    ball = canvas.create_oval(event.x-ball_size, event.y-ball_size, event.x+ball_size, event.y+ball_size, outline='black', fill='red', tags=('ball'))
    points.append((event.x/mapDim["width"], event.y/mapDim["height"]))
    point_params.append((sliders[0].get(), sliders[1].get(), sliders[2].get(), sliders[3].get(), sliders[4].get(), sliders[5].get(), sliders[6].get(), sliders[7].get(), sliders[8].get()))

def train_decoder():
    # Train the model on the points
    if(len(points) < 2):
        print("Not enough points to train")
        return
    print("Training model")
    for epoch in range(100):
        optimizer.zero_grad()
        predicted_params = decoderModel(torch.tensor(points))
        loss = criterion(predicted_params, torch.tensor(point_params))
        loss.backward()
        optimizer.step()
        if(epoch==99):
            print("Final loss="+str(loss.item()))
    print("Training complete")
    

def interpolate(event):
    if(is_predicting):
        predicted_params = decoderModel(torch.tensor([event.x/mapDim["width"], event.y/mapDim["height"]]))
        predicted_params = predicted_params.tolist()
        update_params_from_point(predicted_params)

def simulate_current_state(dt, ball):
    if(is_simulating): #ensures this doesn't loop after sim_ball deletion
        current_state["x"] += current_state["v_x"]*dt
        current_state["y"] += current_state["v_y"]*dt

        margin = 0.015
        if(current_state["x"] > 1-margin or current_state["x"] < margin):
            current_state["x"] = 1-margin if current_state["x"] > 1-margin else margin
            current_state["v_x"] *= -1
        if(current_state["y"] > 1-margin or current_state["y"] < margin):
            current_state["y"] = 1-margin if current_state["y"] > 1-margin else margin
            current_state["v_y"] *= -1

        canvas.coords(ball, current_state["x"]*mapDim["width"]-ball_size, current_state["y"]*mapDim["height"]-ball_size, current_state["x"]*mapDim["width"]+ball_size, current_state["y"]*mapDim["height"]+ball_size)
        #canvas.move(ball, current_state["v_x"]*mapDim["width"]*dt, current_state["v_y"]*mapDim["height"]*dt)
        predicted_params = decoderModel(torch.tensor([current_state["x"], current_state["y"]]))
        predicted_params = predicted_params.tolist()
        update_params_from_point(predicted_params)
        
        canvas.after(dt, lambda:simulate_current_state(dt, ball))



def save_points(): # Save points to a txt file
    with open("points.txt", "w") as f:
        f.write(str(points))
    with open("point_params.txt", "w") as f:
        f.write(str(point_params))

def load_points(): # Load points from a txt file
    global points, point_params
    clear_points()
    with open("points.txt", "r") as f:
        points = eval(f.read())
        for point in points:
            canvas.create_oval(point[0]*mapDim["width"]-ball_size, point[1]*mapDim["height"]-ball_size, point[0]*mapDim["width"]+ball_size, point[1]*mapDim["height"]+ball_size, outline='black', fill='red', tags=('ball'))
    with open("point_params.txt", "r") as f:
        point_params = eval(f.read())

def clear_points(): # Clear points from the canvas
    global points, point_params
    points = []
    point_params = []
    canvas.delete("ball")

def toggle_mode():
    global is_predicting
    is_predicting = not is_predicting

def toggle_simulation():
    global is_simulating
    is_simulating = not is_simulating
    if(is_simulating):
        current_state["x"] = 0.5
        current_state["y"] = 0.5
        sim_ball = canvas.create_oval(current_state["x"]*mapDim["width"]-ball_size, current_state["y"]*mapDim["height"]-ball_size, current_state["x"]*mapDim["width"]+ball_size, current_state["y"]*mapDim["height"]+ball_size, outline='black', fill='blue', tags=('sim_ball'))
        simulate_current_state(dt=10, ball=sim_ball)
    else:
        canvas.delete("sim_ball")

canvas.bind('<Button-1>', cursor_click)
canvas.bind('<B1-Motion>', interpolate)
canvas.bind('<Button-3>', func=lambda event:play_note())

sliders = init_slider_window(sliderFrame, buttonFrame, parameters, save_points, load_points, clear_points, toggle_mode, train_decoder, toggle_simulation)
scan = OscDataReceive(port=port, address=tStickId + "/*", function=updateInput)

sliderFrame.pack(side=TOP)
map2DFrame.pack(side=LEFT)
buttonFrame.pack()

window.mainloop()