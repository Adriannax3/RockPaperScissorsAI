import os
import sys
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ustawienia TensorFlow
tf.config.run_functions_eagerly(True)  # Opcjonalne – przydatne do debugowania

# Ścieżki i parametry
model_path = "hand_gesture_model.h5"
correction_path = "corrections"
image_size = (150, 150)
batch_size = 16
epochs = 5

# Wczytaj model
if not os.path.exists(model_path):
    print(f"❌ Nie znaleziono modelu: {model_path}")
    sys.exit()

print("📦 Wczytywanie modelu...")
model = tf.keras.models.load_model(model_path)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Przygotowanie danych z poprawek
print("📁 Przygotowanie danych z katalogu:", correction_path)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    correction_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Sprawdzenie liczby próbek
if train_data.samples < 3 or train_data.num_classes < 2:
    print("❌ Za mało danych do treningu.")
    print(f"🔍 Znaleziono {train_data.samples} próbek w {train_data.num_classes} klasach.")
    sys.exit()

# Trening
print(f"🧠 Rozpoczynanie douczania modelu na {train_data.samples} próbkach...")
history = model.fit(
    train_data,
    epochs=epochs,
    steps_per_epoch=max(1, train_data.samples // batch_size),
    verbose=1
)

# Zapis modelu
model.save(model_path)
print("✅ Model został douczony i zapisany jako:", model_path)
