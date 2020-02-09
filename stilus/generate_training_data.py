from midi.utils import convert_midi_to_time_series, get_training_data 
from mido import Message, MidiFile, MidiTrack
import os
import numpy as np

path = "../midi/test/"
result = None

for root, dirs, files in os.walk(path):

    for file in files :
        abs_path = os.path.join(root, file)
        print(abs_path)
        mid = MidiFile(abs_path)

        time_series = convert_midi_to_time_series(mid, 5, 5, 8)
        training_data = get_training_data(time_series, 33)

        #print(training_data)
        if result is None :
            result = training_data
        else :
            result = np.vstack((result, training_data))

print(result.shape)
np.save("../test_data", result)

