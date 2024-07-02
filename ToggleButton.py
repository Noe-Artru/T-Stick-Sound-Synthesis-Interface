import tkinter as tk

class ToggleButton:
    def __init__(self, root, on_text="ON", off_text="OFF", extra_command=None):
        self.root = root
        self.button_state = False
        self.on_text = on_text
        self.off_text = off_text
        self.extra_command = extra_command

        self.button = tk.Button(root, text=off_text, command=self.toggle)

    def toggle(self):
        if self.button_state:
            self.button.config(text=self.off_text)
            self.button_state = False
        else:
            self.button.config(text=self.on_text)
            self.button_state = True

        if self.extra_command:
            self.extra_command()