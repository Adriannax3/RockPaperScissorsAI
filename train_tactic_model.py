import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Mapowanie gestów
gesture_to_index = {'rock': 0, 'paper': 1, 'scissors': 2}
index_to_gesture = {v: k for k, v in gesture_to_index.items()}
sequence_length = 2

# 📊 Przykładowy dataset gracza (symulowane ruchy)
# Załóżmy, że gracz często po "rock" gra "paper"
sample_history = [
    'rock', 'paper', 'rock', 'paper', 'rock', 'paper',
    'scissors', 'rock', 'scissors', 'rock',
    'paper', 'scissors', 'paper', 'scissors',
    'rock', 'rock', 'paper', 'scissors'
]

# 🔁 Przygotowanie danych: sekwencje i etykiety
X = []
y = []

for i in range(len(sample_history) - sequence_length):
    seq = sample_history[i:i + sequence_length]
    target = sample_history[i + sequence_length]

    seq_encoded = [gesture_to_index[g] for g in seq]
    target_encoded = gesture_to_index[target]

    X.append(seq_encoded)
    y.append(target_encoded)

X = np.array(X)
y = to_categorical(y, num_classes=3)

# 📦 Podział na trening i test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Budowa modelu
model = Sequential([
    Embedding(input_dim=3, output_dim=10, input_length=sequence_length),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔁 Trening
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=4, verbose=1)

# 💾 Zapis modelu
model.save("tactic_model.h5")
print("✅ Model został zapisany jako tactic_model.h5")
