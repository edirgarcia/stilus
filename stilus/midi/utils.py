import random
import math
import numpy as np
from mido import Message, MidiFile, MidiTrack

def get_total_beats(mid):
    max_ticks = 0
    for i, track in enumerate(mid.tracks):
        total_ticks = 0
        #print('Track {}: {}'.format(i, track.name))
        for msg in track:
            if msg.type == "note_on" :
                total_ticks = total_ticks + msg.time
        if total_ticks> max_ticks:
           max_ticks = total_ticks 
        #print("total ticks:" , total_ticks)
    #print("max ticks: ", max_ticks)
    total_beats = math.ceil(max_ticks / mid.ticks_per_beat)
    return total_beats


### params #mid is is the midi file, 
### max_sim_notes is the maximum amount of simultaneous notes on a track, as of now most pieces have at most 4
### max_granularity is what is the minumum space that you can represent
### 1 is crotchets/quarters, 2 is quavers, 4 is semiquavers, etc. 4,or 8 is is a good default

### returns, tensor - the tensor that processed the whole midi only on the tracks that have note_on events
def convert_midi_to_tensor(mid, max_sim_notes, max_granularity):
    total_beats = get_total_beats(mid)
    tensor = np.zeros((len(mid.tracks), max_sim_notes, total_beats * max_granularity))
    note_on_dims = []
    for i, track in enumerate(mid.tracks):
            total_ticks = 0
            #print('Track {}: {}'.format(i, track.name))
            secondary_index = 0
            prev_index = -1
            note_on_in_track = False
            for msg in track:
                if msg.type == "note_on" :
                    note_on_in_track = True
                    total_ticks = total_ticks + msg.time
                    if msg.velocity > 0 :
                        #print(total_ticks * max_granularity / mid.ticks_per_beat)
                        curr_index = round(total_ticks * max_granularity / mid.ticks_per_beat)
                        #print(curr_index)
                        if prev_index == curr_index :
                            secondary_index = secondary_index + 1
                        else:
                            secondary_index = 0
                        prev_index = curr_index
                        tensor[i, secondary_index, curr_index ] = msg.note
                        #print(curr_index, secondary_index)
            
            if note_on_in_track :
                note_on_dims.append(i)
    #print ("time series shape:", tensor.shape)
    return tensor[note_on_dims]

def delete_all_note_on(mid):
    for i, track in enumerate(mid.tracks):
        j = 0
        while (j < len(track)) :
            if track[j].type == "note_on":
                del track[j]
            else:
                j+=1
    return mid

# Replaces all the note_on events with events from a tensor representation
def convert_tensor_to_midi(original_mid, tensor):
    original_mid = delete_all_note_on(original_mid)
    ### To be implemented
    return None


### params #mid is is the midi file, 
### max_sim_notes is the maximum amount of simultaneous notes on all tracks, as of now most pieces have at most 6
### max_sim_notes_per_track is the maximum amount of simultaneous notes on a track, as of now most pieces have at most 4
### max_granularity is what is the minumum space that you can represent
### 1 is crotchets/quarters, 2 is quavers, 4 is semiquavers, etc. 4,or 8 is is a good default

### returns, time_series - the tensor thatrepresents the sorted notes on all tracks, per time step
def convert_midi_to_time_series(mid, max_sim_notes, max_sim_notes_per_track, max_granularity) :
    tensor = convert_midi_to_tensor(mid, max_sim_notes_per_track, max_granularity)
    #print(tensor.shape)
    all_tracks = np.concatenate(tensor[:], axis=1)
    return  all_tracks

### Generates records of size record_size, from the complete timeseries of a midi
def get_training_data(time_series, record_size):
    result = np.zeros((time_series.shape[1]+1 - record_size, time_series.shape[0], record_size))
    #print(result.shape)
    idx = 0
    time_series_len = time_series.shape[1]
    while idx <= time_series_len - record_size:
        result[idx] = time_series[:,idx:idx+record_size]
        idx = idx + 1
    return result