import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# Ustawienia
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = 'data'  # Folder z podfolderami: rock/, paper/, scissors/

# Wczytywanie i przygotowanie danych
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

# Wczytanie gotowego modelu bazowego
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Zamrożenie wag pretrenowanego modelu
base_model.trainable = False

# Dodanie własnych warstw klasyfikujących
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)  # 3 klasy

model = Model(inputs=base_model.input, outputs=predictions)

# Kompilacja
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Trenowanie
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Zapis modelu
model.save('hand_gesture_model.h5')
print("✅ Model zapisany jako 'hand_gesture_model.h5'")
