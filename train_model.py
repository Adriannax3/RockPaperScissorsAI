import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== KONFIGURACJA ==========
IMAGE_SIZE = (150, 150)          # Rozmiar obrazu wejściowego
BATCH_SIZE = 16                  # Wielkość batcha
EPOCHS = 30                      # Maksymalna liczba epok
INITIAL_LR = 0.0001               # Learning rate
DROPOUT_RATE = 0.2               # Dropout rate
DENSE_LAYERS = [128, 64]         # Liczba neuronów w warstwach gęstych (możesz zmieniać np. na [256, 128] lub [64])
DATA_DIR = 'data'                # Folder z danymi
USE_FINE_TUNING = True          # Czy użyć fine-tuningu (True/False)
FINE_TUNE_AT = 100               # Od której warstwy fine-tuning

# ========== AUTOMATYCZNE KONFIGUROWANIE ==========
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ========== PRZYGOTOWANIE DANYCH ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ========== BUDOWA MODELU ==========
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = not USE_FINE_TUNING  # Zamrażanie warstw jeśli nie używamy fine-tuningu

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Dynamiczne dodawanie warstw na podstawie konfiguracji
for units in DENSE_LAYERS:
    x = Dense(units, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

predictions = Dense(3, activation='softmax')(x)  # 3 klasy wyjściowe
model = Model(inputs=base_model.input, outputs=predictions)

# ========== FINE-TUNING (OPCJONALNIE) ==========
if USE_FINE_TUNING:
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    print(f"\n🔧 Fine-tuning od warstwy {FINE_TUNE_AT} (liczba zamrożonych warstw: {FINE_TUNE_AT}/{len(base_model.layers)})")

# ========== KOMPILACJA ==========
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# ========== CALLBACKS ==========
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    CSVLogger('reports/training_log.csv'),
    TensorBoard(log_dir='logs', histogram_freq=1)
]

# ========== TRENOWANIE ==========
print("\n⚙️ Konfiguracja modelu:")
print(f"- Warstwy gęste: {DENSE_LAYERS}")
print(f"- Learning rate: {INITIAL_LR}")
print(f"- Dropout: {DROPOUT_RATE}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Epochs: {EPOCHS}\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# ========== RAPORT ==========
# Generowanie wykresów
plt.figure(figsize=(15, 5))

metrics = ['accuracy', 'loss', 'precision', 'recall']
for i, metric in enumerate(metrics):
    plt.subplot(1, 4, i+1)
    plt.plot(history.history[metric], label=f'Train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
    plt.title(metric.capitalize())
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend()

plt.tight_layout()
plt.savefig('reports/training_metrics.png')
plt.close()

# Macierz pomyłek i raport klasyfikacji
val_generator.reset()
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Raport klasyfikacji
class_report = classification_report(y_true, y_pred_classes, 
                                   target_names=val_generator.class_indices.keys())

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('reports/confusion_matrix.png')
plt.close()

# Zapis pełnego raportu
with open('reports/training_report.txt', 'w') as f:
    f.write("=== PARAMETRY TRENOWANIA ===\n")
    f.write(f"Model: MobileNetV2\n")
    f.write(f"Warstwy gęste: {DENSE_LAYERS}\n")
    f.write(f"Learning rate: {INITIAL_LR}\n")
    f.write(f"Dropout: {DROPOUT_RATE}\n")
    f.write(f"Rozmiar obrazu: {IMAGE_SIZE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epoki: {EPOCHS}\n")
    f.write(f"Fine-tuning: {'Tak' if USE_FINE_TUNING else 'Nie'}\n")
    if USE_FINE_TUNING:
        f.write(f"Fine-tuning od warstwy: {FINE_TUNE_AT}\n")
    f.write(f"\nNajlepsza dokładność walidacyjna: {max(history.history['val_accuracy']):.4f}\n\n")
    
    f.write("=== RAPORT KLASYFIKACJI ===\n")
    f.write(class_report)

# ========== ZAPIS MODELU ==========
model.save('hand_gesture_model.h5')
print("\n✅ Model zapisany jako 'hand_gesture_model.h5'")
print("📊 Raporty zapisane w folderze 'reports/':")
print("   - training_log.csv (pełne logi)")
print("   - training_metrics.png (wykresy)")
print("   - confusion_matrix.png (macierz pomyłek)")
print("   - training_report.txt (podsumowanie)")
print("📈 Uruchom TensorBoard komendą: tensorboard --logdir logs/")