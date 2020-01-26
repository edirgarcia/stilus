from midi_utils import convert_midi_to_time_series, get_training_data 
from mido import Message, MidiFile, MidiTrack
import os
import numpy as np

path = "../midi/training/"
result = None

for file in os.listdir(path):

    #import a midi file
    abs_path = os.path.join(path, file)
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

