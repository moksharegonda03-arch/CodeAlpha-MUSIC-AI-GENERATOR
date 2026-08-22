import os
from music21 import converter, instrument, note, chord

data_path = "data"
notes = []

for file in os.listdir(data_path):
    if file.endswith(".mid"):
        midi = converter.parse(os.path.join(data_path, file))
        
        parts = instrument.partitionByInstrument(midi)
        
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total notes extracted:", len(notes))