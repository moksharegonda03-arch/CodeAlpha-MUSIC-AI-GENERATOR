import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Load processed music notes
with open("notes_data.pkl", "rb") as f:
    notes = pickle.load(f)

print("Total notes:", len(notes))

# Find unique notes
unique_notes = sorted(set(notes))
n_vocab = len(unique_notes)

print("Unique notes:", n_vocab)

# Convert notes to numbers
note_to_int = {
    note: number
    for number, note in enumerate(unique_notes)
}

int_to_note = {
    number: note
    for note, number in note_to_int.items()
}

# Save mappings
with open("note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

with open("int_to_note.pkl", "wb") as f:
    pickle.dump(int_to_note, f)


# Convert notes to integers
sequence = [
    note_to_int[note]
    for note in notes
]

sequence_length = 50

network_input = []
network_output = []

for i in range(len(sequence) - sequence_length):

    network_input.append(
        sequence[i:i + sequence_length]
    )

    network_output.append(
        sequence[i + sequence_length]
    )


# Convert to NumPy arrays
network_input = np.array(network_input)
network_output = np.array(network_output)

print("Training patterns:", len(network_input))


# Reshape input for LSTM
network_input = network_input.reshape(
    network_input.shape[0],
    network_input.shape[1],
    1
)

# Normalize
network_input = network_input / float(n_vocab)

# One-hot encode output
network_output = to_categorical(
    network_output,
    num_classes=n_vocab
)


# Build LSTM model
model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(sequence_length, 1),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(256)
)

model.add(Dropout(0.3))

model.add(
    Dense(n_vocab, activation="softmax")
)


# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)


# Show model
model.summary()


# Train model
print("Starting training...")

model.fit(
    network_input,
    network_output,
    epochs=30,
    batch_size=64
)


# Save trained model
model.save("music_model.h5")

print("================================")
print("Training completed successfully!")
print("music_model.h5 created!")
print("================================")