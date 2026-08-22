import numpy as np
import pickle
import tensorflow as tf
from music21 import stream, note, chord

print("================================")
print("     MUSIC GENERATION")
print("================================")

# ==============================
# Load trained model
# ==============================
print("Loading trained model...")

model = tf.keras.models.load_model(
    "music_model.h5",
    compile=False
)

print("Model loaded successfully!")

# ==============================
# Load notes
# ==============================
print("Loading music data...")

with open("notes_data.pkl", "rb") as f:
    notes = pickle.load(f)

print("Notes loaded successfully!")
print("Total notes:", len(notes))

# ==============================
# Create note mappings
# ==============================
unique_notes = sorted(set(notes))

note_to_int = {
    note_name: number
    for number, note_name in enumerate(unique_notes)
}

int_to_note = {
    number: note_name
    for number, note_name in enumerate(unique_notes)
}

print("Unique notes:", len(unique_notes))

# ==============================
# Save mappings
# ==============================
with open("note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

with open("int_to_note.pkl", "wb") as f:
    pickle.dump(int_to_note, f)

print("Note mappings created!")

# ==============================
# Check model input
# ==============================
print("Model input shape:", model.input_shape)

sequence_length = model.input_shape[1]

print("Sequence length:", sequence_length)

# ==============================
# Starting pattern
# ==============================
start = np.random.randint(
    0,
    len(notes) - sequence_length
)

pattern = notes[start:start + sequence_length]

generated_notes = []

print("Generating music...")

# ==============================
# Generate 300 notes
# ==============================
for i in range(300):

    input_sequence = [
        note_to_int[n]
        for n in pattern
    ]

    input_sequence = np.reshape(
        input_sequence,
        (1, sequence_length, 1)
    )

    input_sequence = input_sequence / float(
        len(unique_notes)
    )

    prediction = model.predict(
        input_sequence,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    generated_notes.append(result)

    pattern.append(result)
    pattern = pattern[1:]

    if (i + 1) % 50 == 0:
        print("Generated", i + 1, "notes...")

print("Music generation completed!")

# ==============================
# Create MIDI
# ==============================
print("Creating MIDI file...")

output = stream.Stream()

for pattern_note in generated_notes:

    # Chord
    if "." in str(pattern_note):

        try:
            notes_in_chord = pattern_note.split(".")

            chord_notes = [
                note.Note(int(n))
                for n in notes_in_chord
            ]

            new_chord = chord.Chord(chord_notes)

            output.append(new_chord)

        except:
            pass

    # Single note
    else:

        try:
            new_note = note.Note(pattern_note)
            output.append(new_note)

        except:
            pass

# ==============================
# Save MIDI
# ==============================
output_file = "output.mid"

output.write(
    "midi",
    fp=output_file
)

print("================================")
print(" MUSIC GENERATED SUCCESSFULLY!")
print("================================")
print("File created:", output_file)
print("Notes generated:", len(generated_notes))
print("================================")