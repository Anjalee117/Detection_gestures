"""A fixed-layout camera viewport: image dimensions never resize its container."""
import tkinter as tk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk


def fit_frame(frame, width, height):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Preview requires a uint8 BGR frame with three channels")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Preview frame is empty")
    width, height = max(1, int(width)), max(1, int(height))
    scale = min(width / frame.shape[1], height / frame.shape[0])
    size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
    return cv.cvtColor(cv.resize(frame, size, interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)


class VideoPreview(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=1, height=1, highlightthickness=0, **kwargs)
        self._frame = None
        self._photo = None
        self.rendered_frames = 0
        self._image_item = self.create_image(0, 0, anchor="center", state="hidden")
        self._text_item = self.create_text(0, 0, text="WAITING FOR CAMERA",
                                          fill="#C9F264", font=("Arial", 16, "bold"), justify="center")
        self.bind("<Configure>", self._resize)

    def _resize(self, event):
        self.coords(self._text_item, event.width / 2, event.height / 2)
        self.itemconfigure(self._text_item, width=max(1, event.width - 32))
        if self._frame is not None:
            self._draw()

    def show_message(self, message):
        self._frame = self._photo = None
        self.itemconfigure(self._image_item, image="", state="hidden")
        self.itemconfigure(self._text_item, text=message, state="normal")

    def show_frame(self, frame):
        self._frame = frame
        self._draw()

    def _draw(self):
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return  # Configure redraws the retained frame when the dashboard is mapped.
        rgb = fit_frame(self._frame, width, height)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self)
        self.coords(self._image_item, width / 2, height / 2)
        self.itemconfigure(self._image_item, image=self._photo, state="normal")
        self.itemconfigure(self._text_item, state="hidden")
        self.rendered_frames += 1
