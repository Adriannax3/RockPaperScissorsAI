import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienia TensorFlow
tf.config.run_functions_eagerly(True)  # Opcjonalne – przydatne do debugowania

# Ścieżki i parametry
model_path = "hand_gesture_model.h5"
correction_path = "corrections"
report_dir = "training_report"
image_size = (150, 150)
batch_size = 16
epochs = 5

# Utwórz folder na raporty
os.makedirs(report_dir, exist_ok=True)

# Wczytaj model
if not os.path.exists(model_path):
    print(f"❌ Nie znaleziono modelu: {model_path}")
    sys.exit()

print("📦 Wczytywanie modelu...")
model = tf.keras.models.load_model(model_path)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Przygotowanie danych
print("📁 Przygotowanie danych z katalogu:", correction_path)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Dodany podział na walidację

train_data = datagen.flow_from_directory(
    correction_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_data = datagen.flow_from_directory(
    correction_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Sprawdzenie danych
if train_data.samples < 3 or train_data.num_classes < 2:
    print("❌ Za mało danych do treningu.")
    print(f"🔍 Znaleziono {train_data.samples} próbek w {train_data.num_classes} klasach.")
    sys.exit()

# Funkcje do generowania raportu
def save_training_plot(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(f'{report_dir}/training_history.png')
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{report_dir}/confusion_matrix.png')
    plt.close()

# Trening
print(f"🧠 Rozpoczynanie douczania modelu na {train_data.samples} próbkach...")
start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    steps_per_epoch=max(1, train_data.samples // batch_size),
    validation_steps=max(1, val_data.samples // batch_size),
    verbose=1
)

training_time = time.time() - start_time

# Generowanie raportu
print("📊 Generowanie raportu...")

# 1. Wykresy treningu
save_training_plot(history)

# 2. Macierz pomyłek i metryki
val_data.reset()
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes

save_confusion_matrix(y_true, y_pred_classes, list(val_data.class_indices.keys()))

report = classification_report(y_true, y_pred_classes, 
                              target_names=list(val_data.class_indices.keys()),
                              output_dict=True)

# 3. Zapis pełnego raportu
with open(f'{report_dir}/training_report.txt', 'w') as f:
    f.write("=== PODSUMOWANIE DOUCZANIA ===\n\n")
    f.write(f"Czas treningu: {training_time:.2f}s\n")
    f.write(f"Liczba epok: {epochs}\n")
    f.write(f"Liczba próbek treningowych: {train_data.samples}\n")
    f.write(f"Liczba próbek walidacyjnych: {val_data.samples}\n\n")
    
    f.write("=== NAJLEPSZE WYNIKI ===\n")
    f.write(f"Val Accuracy: {max(history.history['val_accuracy']):.4f}\n")
    f.write(f"Val Loss: {min(history.history['val_loss']):.4f}\n\n")
    
    f.write("=== RAPORT KLASYFIKACJI ===\n")
    f.write(classification_report(y_true, y_pred_classes, 
                                 target_names=list(val_data.class_indices.keys())))
    
    f.write("\n=== HISTORIA TRENINGU ===\n")
    for epoch in range(epochs):
        f.write(f"Epoch {epoch+1}: ")
        f.write(f"acc={history.history['accuracy'][epoch]:.4f}, ")
        f.write(f"val_acc={history.history['val_accuracy'][epoch]:.4f}, ")
        f.write(f"loss={history.history['loss'][epoch]:.4f}, ")
        f.write(f"val_loss={history.history['val_loss'][epoch]:.4f}\n")

# Zapis modelu
model.save(model_path)
print(f"\n✅ Model został douczony i zapisany jako: {model_path}")
print(f"📂 Pełny raport dostępny w folderze: {report_dir}/")
print("Zawartość raportu:")
print(f"- {report_dir}/training_history.png (wykresy accuracy i loss)")
print(f"- {report_dir}/confusion_matrix.png (macierz pomyłek)")
print(f"- {report_dir}/training_report.txt (szczegółowe metryki)")