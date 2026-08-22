import pickle
import numpy as np

from tensorflow.keras.models import load_model
from music21 import stream, note, chord


# -----------------------------
# File names
# -----------------------------

MODEL_FILE = "music_model.h5"
NOTES_FILE = "notes_data.pkl"
NOTE_TO_INT_FILE = "note_to_int.pkl"
INT_TO_NOTE_FILE = "int_to_note.pkl"
OUTPUT_FILE = "output.mid"


# -----------------------------
# Generate music
# -----------------------------

def generate_music(number_of_notes=200):

    print("================================")
    print("      AI MUSIC GENERATOR")
    print("================================")

    # Load trained model
    model = load_model(MODEL_FILE)

    # Load original notes
    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    # Load note mappings
    with open(NOTE_TO_INT_FILE, "rb") as f:
        note_to_int = pickle.load(f)

    with open(INT_TO_NOTE_FILE, "rb") as f:
        int_to_note = pickle.load(f)

    # Sequence length used during training
    sequence_length = 50

    # Convert notes to integers
    encoded_notes = [
        note_to_int[n]
        for n in notes
    ]

    # Select random starting point
    start = np.random.randint(
        0,
        len(encoded_notes) - sequence_length - 1
    )

    pattern = encoded_notes[
        start:start + sequence_length
    ]

    prediction_output = []

    print("Generating music...")

    # Generate notes
    for _ in range(number_of_notes):

        prediction_input = np.reshape(
            pattern,
            (1, sequence_length, 1)
        )

        prediction_input = (
            prediction_input /
            float(len(note_to_int))
        )

        prediction = model.predict(
            prediction_input,
            verbose=0
        )

        # Select predicted note
        index = np.argmax(prediction)

        result = int_to_note[index]

        prediction_output.append(result)

        # Move sequence forward
        pattern.append(index)
        pattern = pattern[1:]

    # -----------------------------
    # Convert predictions to MIDI
    # -----------------------------

    output_notes = []

    for pattern in prediction_output:

        # Chord
        if "." in pattern:

            try:

                numbers = [
                    int(n)
                    for n in pattern.split(".")
                ]

                new_chord = chord.Chord(numbers)

                output_notes.append(new_chord)

            except Exception:
                continue

        # Single note
        else:

            try:

                new_note = note.Note(pattern)

                output_notes.append(new_note)

            except Exception:
                continue

    # Create MIDI stream
    midi_stream = stream.Stream()

    for element in output_notes:
        midi_stream.append(element)

    # -----------------------------
    # SAVE OUTPUT MIDI
    # -----------------------------

    midi_stream.write(
        "midi",
        fp=OUTPUT_FILE
    )

    print("================================")
    print("Music generated successfully!")
    print("Saved as:", OUTPUT_FILE)
    print("================================")

    return OUTPUT_FILE


# -----------------------------
# Run program
# -----------------------------

if __name__ == "__main__":
    generate_music(200)