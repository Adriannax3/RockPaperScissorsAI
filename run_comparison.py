import os
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== DEFINICJE KONFIGURACJI ==========
CONFIGS = [
    {
        "name": "Basic",
        "image_size": (150, 150),
        "batch_size": 32,
        "epochs": 15,
        "lr": 0.001,
        "dropout": 0.3,
        "dense_layers": [128],
        "fine_tune": False
    },
    {
        "name": "Deep",
        "image_size": (150, 150),
        "batch_size": 32,
        "epochs": 25,
        "lr": 0.0005,
        "dropout": 0.4,
        "dense_layers": [256, 128, 64],
        "fine_tune": False
    },
    {
        "name": "Fine-Tune",
        "image_size": (150, 150),
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.0001,
        "dropout": 0.2,
        "dense_layers": [128, 64],
        "fine_tune": True,
        "fine_tune_at": 100
    },
    {
        "name": "High-Accuracy",
        "image_size": (224, 224),
        "batch_size": 16,
        "epochs": 40,
        "lr": 0.00005,
        "dropout": 0.5,
        "dense_layers": [256, 128, 64],
        "fine_tune": True,
        "fine_tune_at": 80
    },
    {
        "name": "Fast-Training",
        "image_size": (128, 128),
        "batch_size": 64,
        "epochs": 10,
        "lr": 0.01,
        "dropout": 0.2,
        "dense_layers": [64],
        "fine_tune": False
    }
]

# ========== FUNKCJE POMOCNICZE ==========
def create_model(config, num_classes=3):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(config['image_size'][0], config['image_size'][1], 3)
    )
    base_model.trainable = not config['fine_tune']

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    for units in config['dense_layers']:
        x = Dense(units, activation='relu')(x)
        x = Dropout(config['dropout'])(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    if config['fine_tune']:
        base_model.trainable = True
        for layer in base_model.layers[:config['fine_tune_at']]:
            layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=config['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_training_history(history, config_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'comparison_results/{config_name}/training_history.png')
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, config_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'comparison_results/{config_name}/confusion_matrix.png')
    plt.close()

def train_and_evaluate(config, data_dir='data'):
    os.makedirs(f"comparison_results/{config['name']}", exist_ok=True)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation'
    )
    
    model = create_model(config)
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=config['epochs'],
        validation_data=val_generator,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    val_generator.reset()
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes
    
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    
    save_training_history(history, config['name'])
    save_confusion_matrix(y_true, y_pred_classes, list(val_generator.class_indices.keys()), config['name'])
    
    return {
        'config_name': config['name'],
        'val_accuracy': max(history.history['val_accuracy']),
        'final_val_loss': history.history['val_loss'][-1],
        'training_time': training_time,
        'overfitting': max(history.history['accuracy']) - max(history.history['val_accuracy']),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }

# ========== GŁÓWNA CZĘŚĆ SKRYPTU ==========
if __name__ == "__main__":
    os.makedirs("comparison_results", exist_ok=True)
    results = []
    
    print("🚀 Rozpoczynam testowanie konfiguracji...")
    for config in CONFIGS:
        print(f"\n🔍 Testowanie konfiguracji: {config['name']}")
        result = train_and_evaluate(config)
        results.append(result)
        print(f"✅ Zakończono: Val Accuracy = {result['val_accuracy']:.4f}")
        print(f"   Wykresy zapisane w: comparison_results/{config['name']}/")
    
    with open('comparison_results/comparison_report.csv', 'w', newline='') as csvfile:
        fieldnames = ['config_name', 'val_accuracy', 'final_val_loss', 'training_time', 
                     'overfitting', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    metrics = ['val_accuracy', 'final_val_loss', 'training_time', 'overfitting']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.bar([r['config_name'] for r in results], [r[metric] for r in results])
        plt.title(metric.replace('_', ' ').capitalize())
        if metric == 'training_time':
            plt.ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig('comparison_results/comparison_chart.png')
    plt.close()
    
    print("\n📊 Wyniki zapisane w folderze 'comparison_results/':")
    print("   - comparison_report.csv (pełne dane)")
    print("   - comparison_chart.png (wizualizacja)")