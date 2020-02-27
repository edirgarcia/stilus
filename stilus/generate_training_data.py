from midi.utils import convert_midi_to_time_series, get_training_data 
from mido import Message, MidiFile, MidiTrack
from sklearn.model_selection import train_test_split
from sys import argv
import os
import numpy as np

# <your_path>\Source\Repos\stilus> python stilus/generate_training_data.py midi/test/ 
#                                  python stilus/generate_training_data.py midi/training/ 

if len(argv) != 2:
    raise ValueError("This script must have 2 parameters inputPath, and outputPath")


print("Generating training data...")

script_name, in_path = argv
# in_path = "../midi/training/"
# out_path = "../training_data"

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


train, val = train_test_split(result, test_size=.15, random_state=42, shuffle=True)
print("train resulting tensor shape:", train.shape)
print("val resulting tensor shape:", val.shape)
np.save("training_data", train)
np.save("validation_data", val)

