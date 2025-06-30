import tensorflow as tf
tf.config.run_functions_eagerly(True)

import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import random
import threading
import time
import os
from tensorflow.keras.optimizers import Adam

# === Wczytanie modeli ===
gesture_model = load_model("hand_gesture_model.h5")

# Wczytanie lub utworzenie modelu taktycznego
TACTIC_MODEL_PATH = "tactic_model.h5"
sequence_length = 2
gesture_to_index = {'rock': 0, 'paper': 1, 'scissors': 2}
index_to_gesture = {v: k for k, v in gesture_to_index.items()}

if os.path.exists(TACTIC_MODEL_PATH):
    tactic_model = load_model(TACTIC_MODEL_PATH)
    print("✅ Wczytano tactic_model.h5")
    
    tactic_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
else:
    tactic_model = Sequential([
        Embedding(input_dim=3, output_dim=10, input_length=sequence_length),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    tactic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("🧠 Utworzono nowy model taktyczny")

# Historia ruchów gracza
user_history = []

# Funkcja: przewiduj przyszły ruch gracza
def predict_user_next_move():
    if len(user_history) < sequence_length:
        return random.choice(list(gesture_to_index.keys()))

    seq = user_history[-sequence_length:]
    seq_encoded = np.array([[gesture_to_index[move] for move in seq]])
    pred = tactic_model.predict(seq_encoded, verbose=0)[0]
    predicted_index = np.argmax(pred)
    return index_to_gesture[predicted_index]

# Funkcja: wybór ruchu AI, który bije przewidywany ruch gracza
def counter_move(predicted_user_move):
    beats = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return beats.get(predicted_user_move, random.choice(list(gesture_to_index.keys())))

# Funkcja: aktualizacja modelu taktycznego na podstawie ruchów gracza
def retrain_tactic_model():
    if len(user_history) <= sequence_length:
        return

    X, y = [], []
    for i in range(len(user_history) - sequence_length):
        seq = user_history[i:i + sequence_length]
        target = user_history[i + sequence_length]
        X.append([gesture_to_index[m] for m in seq])
        y.append(gesture_to_index[target])

    X = np.array(X)
    y = to_categorical(y, num_classes=3)

    tactic_model.fit(X, y, epochs=1, verbose=0)
    tactic_model.save(TACTIC_MODEL_PATH)

# === Funkcja predykcji gestu ===
def predict_gesture(image):
    image = cv2.resize(image, (150, 150)) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = gesture_model.predict(image, verbose=0)[0]
    idx = np.argmax(preds)
    return index_to_gesture[idx], float(preds[idx]) * 100

def find_working_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return i
    return -1

# === Mapy wyników ===
outcome_map = {
    ("rock", "scissors"): "WYGRAŁEŚ",
    ("rock", "paper"): "PRZEGRAŁEŚ",
    ("paper", "rock"): "WYGRAŁEŚ",
    ("paper", "scissors"): "PRZEGRAŁEŚ",
    ("scissors", "paper"): "WYGRAŁEŚ",
    ("scissors", "rock"): "PRZEGRAŁEŚ",
}

# Ikony dla gestów
gesture_icons = {
    "rock": "✊",
    "paper": "🤚",
    "scissors": "✌"
}

# === Główna aplikacja ===
class RPSApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rock Paper Scissors AI")
        self.window.geometry("800x600")
        self.window.resizable(False, False)
        self.window.configure(bg="#f0f0f0")

        # Stylizacja
        self.button_style = {
            "font": ("Arial", 14, "bold"),
            "bg": "#4CAF50",
            "fg": "white",
            "activebackground": "#45a049",
            "borderwidth": 0,
            "highlightthickness": 0,
            "padx": 20,
            "pady": 10
        }

        self.label_style = {
            "font": ("Arial", 16),
            "bg": "#f0f0f0"
        }

        self.result_style = {
            "font": ("Arial", 18, "bold"),
            "bg": "#f0f0f0"
        }

        self.camera_index = find_working_camera()
        if self.camera_index == -1:
            messagebox.showerror("Błąd kamery", "Nie wykryto działającej kamery.")
            window.destroy()
            return

        self.video_capture = cv2.VideoCapture(self.camera_index)

        # Główny kontener
        main_container = tk.Frame(window, bg="#f0f0f0")
        main_container.pack(expand=True, fill="both", padx=20, pady=20)

        # Górna część - widok kamery i AI
        top_frame = tk.Frame(main_container, bg="#f0f0f0")
        top_frame.pack(fill="both", expand=True)

        # Lewa ramka - kamera użytkownika (kwadratowa)
        self.camera_container = tk.Frame(top_frame, width=300, height=300, bg="#f0f0f0")
        self.camera_container.pack_propagate(False)
        self.camera_container.pack(side="left", expand=True)
        
        self.camera_frame = tk.LabelFrame(self.camera_container, text="Twoja kamera", font=("Arial", 12, "bold"), 
                                        bg="white", bd=2, relief="groove", width=300, height=300)
        self.camera_frame.pack_propagate(False)
        self.camera_frame.pack(expand=True, fill="both")
        
        self.camera_label = Label(self.camera_frame, bg="black")
        self.camera_label.pack(expand=True, fill="both")

        # Prawa ramka - ruch AI (kwadratowa)
        self.ai_container = tk.Frame(top_frame, width=300, height=300, bg="#f0f0f0")
        self.ai_container.pack_propagate(False)
        self.ai_container.pack(side="right", expand=True)
        
        self.ai_frame = tk.LabelFrame(self.ai_container, text="Ruch AI", font=("Arial", 12, "bold"), 
                                    bg="white", bd=2, relief="groove", width=300, height=300)
        self.ai_frame.pack_propagate(False)
        self.ai_frame.pack(expand=True, fill="both")
        
        self.ai_display = tk.Canvas(self.ai_frame, bg="white", highlightthickness=0)
        self.ai_display.pack(expand=True, fill="both")
        
        # Tekst na canvasie będzie wyśrodkowany
        self.ai_text = self.ai_display.create_text(150, 150, text="", font=("Arial", 72), fill="black")

        # Dolna część - kontrolki
        bottom_frame = tk.Frame(main_container, bg="#f0f0f0")
        bottom_frame.pack(fill="x", pady=(0, 20))

        # Przycisk start
        self.start_button = Button(bottom_frame, text="ROZPOCZNIJ GRĘ", command=self.start_game, **self.button_style)
        self.start_button.pack(pady=10)

        # Etykieta wyniku
        self.result_label = Label(bottom_frame, text="", **self.result_style)
        self.result_label.pack()

        # Etykieta statusu
        self.status_label = Label(bottom_frame, text="Kliknij przycisk, aby rozpocząć", **self.label_style)
        self.status_label.pack()

        self.update_camera()

    def update_camera(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((300, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.configure(image=imgtk)
            self.camera_label.image = imgtk
        self.window.after(10, self.update_camera)

    def start_game(self):
        self.start_button.config(state="disabled")
        self.status_label.config(text="Przygotuj się...")
        threading.Thread(target=self.countdown_and_capture).start()

    def countdown_and_capture(self):
        for i in range(3, 0, -1):
            self.status_label.config(text=f"Pokazuj swój ruch za {i}...")
            time.sleep(1)
        
        self.status_label.config(text="Pokazuj swój ruch!")
        time.sleep(0.5)

        ret, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        user_move, confidence = predict_gesture(frame)

        user_history.append(user_move)
        retrain_tactic_model()

        predicted_user_move = predict_user_next_move()
        ai_move = counter_move(predicted_user_move)

        # Aktualizacja interfejsu AI - czyszczenie canvasu i dodanie nowego tekstu
        self.ai_display.delete("all")
        self.ai_text = self.ai_display.create_text(150, 150, text=gesture_icons.get(ai_move, ""), 
                                                 font=("Arial", 72), fill="black")

        if user_move == ai_move:
            result = "REMIS"
        else:
            result = outcome_map.get((user_move, ai_move), "PRZEGRAŁEŚ")

        self.result_label.config(
            text=f"Twój ruch: {user_move.upper()} ({confidence:.1f}%)\nWynik: {result}",
            fg="green" if result == "WYGRAŁEŚ" else "red" if result == "PRZEGRAŁEŚ" else "black"
        )

        self.status_label.config(text="Kliknij przycisk, aby zagrać ponownie")
        self.start_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = RPSApp(root)
    root.mainloop()