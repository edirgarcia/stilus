from midi_utils import convert_midi_to_time_series, get_training_data 
from mido import Message, MidiFile, MidiTrack

#import a midi file
mid = MidiFile("../midi/elise.mid")

time_series = convert_midi_to_time_series(mid, 5, 5, 8)
training_data = get_training_data(time_series, 33)

print(training_data)

