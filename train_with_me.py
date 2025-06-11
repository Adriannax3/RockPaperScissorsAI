import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, messagebox, ttk
from PIL import Image, ImageTk
import os
import time
import random
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Klasy i ścieżki
class_names = ['papper', 'rock', 'scissors']
model_path = "hand_gesture_model.h5"
correction_path = "corrections"

# Upewnij się, że foldery istnieją
for cls in class_names:
    os.makedirs(os.path.join(correction_path, cls), exist_ok=True)

# Wczytaj model
model = load_model(model_path)

# Funkcja predykcji
def predict_frame(frame):
    img = cv2.resize(frame, (150, 150)) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    return class_names[idx], float(preds[idx]) * 100

# GUI aplikacji
class TrainerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Trenuj AI z kamerki")
        self.window.geometry("700x500")
        self.video_capture = cv2.VideoCapture(0)

        # Obraz z kamerki
        self.camera_label = Label(window, bg="lightgray")
        self.camera_label.pack(pady=10)

        # Wynik AI
        self.prediction_label = Label(window, text="Czekam na obraz...", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

        # Przycisk – AI dobrze
        self.ok_button = Button(window, text="✅ DOBRZE", font=("Arial", 14), command=self.accept_prediction)
        self.ok_button.pack(pady=5)

        # Przycisk – AI źle + wybór klasy
        self.choice = tk.StringVar()
        self.choice.set(class_names[0])
        self.dropdown = ttk.Combobox(window, values=class_names, textvariable=self.choice, font=("Arial", 12))
        self.dropdown.pack()

        self.fix_button = Button(window, text="❌ TO BYŁO COŚ INNEGO", font=("Arial", 12), command=self.save_correction)
        self.fix_button.pack(pady=5)

        # Trening
        self.train_button = Button(window, text="🧠 TRENUJ PONOWNIE", font=("Arial", 14), command=self.retrain_model)
        self.train_button.pack(pady=10)

        self.status_label = Label(window, text="", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.frame = None
        self.current_prediction = None
        self.update_camera()

    def update_camera(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.frame = frame
            pred, prob = predict_frame(frame)
            self.current_prediction = pred
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(img, (300, 300)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            self.prediction_label.config(text=f"🤖 AI: {pred.upper()} ({prob:.1f}%)")
        self.window.after(100, self.update_camera)

    def accept_prediction(self):
        self.status_label.config(text="✅ Zatwierdzono. Nic nie zapisano.")

    def save_correction(self):
        if self.frame is not None:
            correct_class = self.choice.get()
            timestamp = int(time.time())
            save_path = os.path.join(correction_path, correct_class, f"{timestamp}.jpg")
            cv2.imwrite(save_path, self.frame)
            self.status_label.config(text=f"📸 Zapisano do: {correct_class}/")

    def retrain_model(self):
        self.status_label.config(text="🔄 Trening...")
        self.window.update()

        # Generator do danych z poprawek
        correction_gen = ImageDataGenerator(rescale=1./255)
        correction_data = correction_gen.flow_from_directory(
            correction_path,
            target_size=(150, 150),
            batch_size=16,
            class_mode='categorical',
            shuffle=True
        )

        if correction_data.samples < 3:
            self.status_label.config(text="❌ Za mało danych do treningu.")
            return

        # Wczytaj istniejący model
        base_model = load_model(model_path)

        # Dodatkowe trening tylko na poprawkach (kontynuacja)
        base_model.fit(
            correction_data,
            epochs=5,
            verbose=1
        )

        base_model.save(model_path)
        self.status_label.config(text="✅ Model douczony i zapisany!")

# Uruchom
if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerApp(root)
    root.mainloop()
