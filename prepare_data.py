import os
import pickle
from music21 import converter, instrument, note, chord

DATA_FOLDER = "data"
OUTPUT_FILE = "notes_data.pkl"

notes = []

print("================================")
print("   PREPARING MUSIC DATA")
print("================================")

# Find all MIDI files
midi_files = []

for root, folders, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.lower().endswith((".mid", ".midi")):
            midi_files.append(
                os.path.join(root, file)
            )

print("MIDI files found:", len(midi_files))

# Process every MIDI file
for i, file_path in enumerate(midi_files, 1):

    print(
        f"Processing {i}/{len(midi_files)}: "
        f"{os.path.basename(file_path)}"
    )

    try:
        midi = converter.parse(file_path)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:

            # Single note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            # Chord
            elif isinstance(element, chord.Chord):
                notes.append(
                    ".".join(
                        str(n)
                        for n in element.normalOrder
                    )
                )

    except Exception as e:
        print("Error:", e)

print("--------------------------------")
print("Total notes:", len(notes))
print("Unique notes:", len(set(notes)))

# Save notes as pickle file
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(notes, f)

print("--------------------------------")
print("Data prepared successfully!")
print("Created:", OUTPUT_FILE)
print("================================")