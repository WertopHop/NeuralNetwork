from tkinter import *
from tkinter import ttk
import numpy as np
from typing import List
import time
from neuro import NeuralNetwork

class Paint:
    def __init__(self):
        self.root = Tk()
        self.root.title("Simple Paint 28x28")
        self.CANVAS_SIZE = 280
        self.GRID_SIZE = 28
        self.BRUSH_SIZE = 3
        self.SCALE = self.CANVAS_SIZE // self.GRID_SIZE
        self.is_brush = True
        
        self._setup_ui()
        self._setup_canvas()
        self._init_arrays()
        self.last_update = time.time()
        self.update_array()

    def _setup_ui(self):
        main_frame = Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        left_frame = Frame(main_frame)
        left_frame.pack(side=LEFT, padx=(0, 20))
        controls_frame = Frame(left_frame)
        controls_frame.pack(pady=(0, 10))
        self.tool_button = Button(controls_frame, text="Кисть", width=10, 
                                command=self.toggle_tool, bg='black', fg='white')
        self.tool_button.pack(side=LEFT, padx=(0, 5))
        
        clear_button = Button(controls_frame, text="Очистить", width=10, 
                            command=self.clear_canvas, bg='gray', fg='white')
        clear_button.pack(side=LEFT)
        
        self.canvas = Canvas(left_frame, 
                           width=self.CANVAS_SIZE, 
                           height=self.CANVAS_SIZE, 
                           bg='white')
        self.canvas.pack()
        
        self._setup_progress_bars(main_frame)

    def _setup_progress_bars(self, main_frame: Frame):
        right_frame = Frame(main_frame)
        right_frame.pack(side=LEFT)
        self.style = ttk.Style()
        self.style.configure("Red.Horizontal.TProgressbar", 
                            troughcolor='white', 
                            background='red')
        
        self.progress_bars: List[ttk.Progressbar] = []
        for i in range(10):
            row_frame = Frame(right_frame)
            row_frame.pack(pady=2)
            Label(row_frame, text=str(i), width=2).pack(side=LEFT, padx=(0, 5))
            progress = ttk.Progressbar(row_frame, value=0, length=100, mode='determinate',
                                        style="Red.Horizontal.TProgressbar")
            progress.pack(side=LEFT)
            self.progress_bars.append(progress)

    def _setup_canvas(self):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

    def _init_arrays(self):
        self.pixel_values = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.brush_matrix = self._create_brush_matrix()

    def _create_brush_matrix(self) -> np.ndarray:
        size = self.BRUSH_SIZE * 2 + 1
        y, x = np.ogrid[-self.BRUSH_SIZE:self.BRUSH_SIZE + 1, 
                       -self.BRUSH_SIZE:self.BRUSH_SIZE + 1]
        return np.clip(1 - np.sqrt(x*x + y*y)/self.BRUSH_SIZE, 0, 1)

    def toggle_tool(self):
        self.is_brush = not self.is_brush
        if self.is_brush:
            self.tool_button.config(text="Кисть", bg='black', fg='white')
        else:
            self.tool_button.config(text="Ластик", bg='white', fg='black')

    def clear_canvas(self):
        self.pixel_values = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.canvas.delete("all")

    def paint(self, event):
        px = event.x // self.SCALE
        py = event.y // self.SCALE
        y_start = max(0, py - self.BRUSH_SIZE)
        y_end = min(self.GRID_SIZE, py + self.BRUSH_SIZE + 1)
        x_start = max(0, px - self.BRUSH_SIZE)
        x_end = min(self.GRID_SIZE, px + self.BRUSH_SIZE + 1)
        brush_y_start = y_start - (py - self.BRUSH_SIZE)
        brush_x_start = x_start - (px - self.BRUSH_SIZE)
        brush_y_end = brush_y_start + (y_end - y_start)
        brush_x_end = brush_x_start + (x_end - x_start)
        
        if self.is_brush:
            self.pixel_values[y_start:y_end, x_start:x_end] = np.minimum(1.0, 
                self.pixel_values[y_start:y_end, x_start:x_end] + 
                self.brush_matrix[brush_y_start:brush_y_end, brush_x_start:brush_x_end]
            )
        else:
            self.pixel_values[y_start:y_end, x_start:x_end] = np.maximum(0.0, 
                self.pixel_values[y_start:y_end, x_start:x_end] - 
                self.brush_matrix[brush_y_start:brush_y_end, brush_x_start:brush_x_end]
            )
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                color_value = int(255 - (self.pixel_values[y, x] * 255))
                color = f'#{color_value:02x}{color_value:02x}{color_value:02x}'
                self.canvas.create_rectangle(
                    x * self.SCALE, y * self.SCALE,
                    (x + 1) * self.SCALE, (y + 1) * self.SCALE,
                    fill=color, outline=color
                )

    def update_progress_bar_color(self, progress_bar: ttk.Progressbar, value: float):
        red = int(255 * (1 - value/100))
        green = int(255 * (value/100))
        color = f'#{red:02x}{green:02x}00'
        
        style_name = f"Color{value:.0f}.Horizontal.TProgressbar"
        self.style.configure(style_name, 
                            troughcolor='white',
                            background=color)
        progress_bar.configure(style=style_name)

    def update_array(self):
        current_time = time.time()
        if current_time - self.last_update >= 2.0:
            array = self.get_array().copy()
            neuro = NeuralNetwork()
            data = neuro.feedforward(array)
            for i, bar in enumerate(self.progress_bars):
                value = data[i] * 100
                bar['value'] = value
                self.update_progress_bar_color(bar, value)
            self.last_update = current_time
        self.root.after(100, self.update_array)

    def get_array(self) -> np.ndarray:
        return np.round(self.pixel_values).astype(float)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = Paint()
    app.run()
