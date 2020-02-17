from midi.utils import convert_midi_to_time_series, get_training_data 
from mido import Message, MidiFile, MidiTrack
from sys import argv
import os
import numpy as np

# <your_path>\Source\Repos\stilus> python stilus/generate_training_data.py midi/test/ test_data
#                                  python stilus/generate_training_data.py midi/training/ training_data

print("Generating training data...")
if len(argv) != 3:
    raise ValueError("This script must have 2 parameters inputPath, and outputPath")

script_name, in_path, out_path = argv

result = None

for root, dirs, files in os.walk(in_path):
    for file in files :
        abs_path = os.path.join(root, file)
        print("processing file: ", abs_path)
        mid = MidiFile(abs_path)

        time_series = convert_midi_to_time_series(mid, 5, 5, 8)
        training_data = get_training_data(time_series, 33)

        if result is None :
            result = training_data
        else :
            result = np.vstack((result, training_data))

print("resulting tensor shape:", result.shape)
np.save(out_path, result)

