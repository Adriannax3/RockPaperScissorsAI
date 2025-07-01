import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical

# Konwersja ruchów na liczby
gesture_to_index = {'rock': 0, 'paper': 1, 'scissors': 2}
index_to_gesture = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Przykładowe dane gracza (sekwencja jego ruchów)
player_moves = ['rock', 'paper', 'rock', 'rock', 'scissors', 'paper', 'rock']

# Tworzenie sekwencji (np. 2 poprzednie ruchy => 1 kolejny)
sequence_length = 2
X, y = [], []

for i in range(len(player_moves) - sequence_length):
    seq = player_moves[i:i+sequence_length]
    target = player_moves[i+sequence_length]
    X.append([gesture_to_index[move] for move in seq])
    y.append(gesture_to_index[target])

X = np.array(X)
y = to_categorical(y, num_classes=3)

# Budujemy prosty model LSTM
model = Sequential([
    Embedding(input_dim=3, output_dim=10, input_length=sequence_length),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trening
model.fit(X, y, epochs=100, verbose=0)

# Przykład: ostatnie 2 ruchy gracza
last_moves = ['scissors', 'paper']
input_seq = np.array([[gesture_to_index[move] for move in last_moves]])

# Predykcja kolejnego ruchu gracza
pred = model.predict(input_seq)
predicted_move = np.argmax(pred)
predicted_name = index_to_gesture[predicted_move]

# Wybierz kontr-ruch
def counter_move(move):
    return {
        'rock': 'paper',
        'paper': 'scissors',
        'scissors': 'rock'
    }[move]

counter = counter_move(predicted_name)

print(f"🤖 Przewiduję, że gracz da: {predicted_name}")
print(f"🎯 Więc zagrywam: {counter}")
