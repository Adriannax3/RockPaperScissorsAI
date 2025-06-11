import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import random
import threading
import time

# Wczytanie modelu
model = load_model("hand_gesture_model.h5")
class_names = ['papper', 'rock', 'scissors']  # Dopasuj do etykiet z treningu

outcome_map = {
    ("rock", "scissors"): "WYGRAŁEŚ",
    ("rock", "paper"): "PRZEGRAŁEŚ",
    ("paper", "rock"): "WYGRAŁEŚ",
    ("paper", "scissors"): "PRZEGRAŁEŚ",
    ("scissors", "paper"): "WYGRAŁEŚ",
    ("scissors", "rock"): "PRZEGRAŁEŚ",
}

def predict_gesture(image):
    image = cv2.resize(image, (150, 150)) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]
    idx = np.argmax(preds)
    return class_names[idx], float(preds[idx]) * 100

def get_ai_move():
    return random.choice(class_names)

def find_working_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Kamera nr {i} działa.")
                return i
    return -1

class RPSApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rock Paper Scissors AI")
        self.window.geometry("700x300")
        self.window.resizable(False, False)

        self.camera_index = find_working_camera()
        if self.camera_index == -1:
            messagebox.showerror("Błąd kamery", "Nie wykryto działającej kamery.")
            window.destroy()
            return

        self.video_capture = cv2.VideoCapture(self.camera_index)

        # === LEWA RAMKA: AI ===
        self.left_frame = tk.Frame(window, width=200, height=200, bg="white")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.left_frame.grid_propagate(False)

        self.left_label = Label(self.left_frame, text="AI", font=("Arial", 14), bg="white")
        self.left_label.pack(expand=True)

        # === ŚRODKOWA RAMKA ===
        center_frame = tk.Frame(window)
        center_frame.grid(row=0, column=1, padx=10, pady=10)

        self.center_label = Label(center_frame, text="Kliknij START", font=("Arial", 16))
        self.center_label.pack(pady=10)

        self.start_button = Button(center_frame, text="START", command=self.start_game, font=("Arial", 14), width=10)
        self.start_button.pack()

        self.result_label = Label(center_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # === PRAWA RAMKA: KAMERA ===
        self.right_frame = Label(window, text="Kamerka", width=200, height=200, bg="lightgray")
        self.right_frame.grid(row=0, column=2, padx=10, pady=10)

        self.update_camera()

    def update_camera(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((200, 200))
            imgtk = ImageTk.PhotoImage(image=img)
            self.right_frame.configure(image=imgtk)
            self.right_frame.image = imgtk
        self.window.after(10, self.update_camera)

    def start_game(self):
        self.start_button.config(state="disabled")
        threading.Thread(target=self.countdown_and_capture).start()

    def countdown_and_capture(self):
        for i in range(3, 0, -1):
            self.center_label.config(text=str(i))
            time.sleep(1)
        self.center_label.config(text="✊🤚✌")
        time.sleep(0.5)

        ret, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        user_move, confidence = predict_gesture(frame)

        ai_move = get_ai_move()
        self.left_label.config(text=f"AI: {ai_move.upper()}")

        if user_move == ai_move:
            result = "REMIS"
        else:
            result = outcome_map.get((user_move, ai_move), "PRZEGRAŁEŚ")

        self.result_label.config(
            text=f"Ty: {user_move.upper()} ({confidence:.1f}%)\n{result}",
            fg="green" if result == "WYGRAŁEŚ" else "red" if result == "PRZEGRAŁEŚ" else "black"
        )

        self.start_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = RPSApp(root)
    root.mainloop()
