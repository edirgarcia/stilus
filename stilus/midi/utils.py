import random
import math
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage

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
### 1 is crotchets/quarters, 2 is quavers, 4 is semiquavers, etc. 4, or 8 is is a good default

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

    
### returns, string - the string that processed the whole midi only on the tracks that have note_on events
def convert_midi_to_string(mid, max_sim_notes, max_granularity):
    total_beats = get_total_beats(mid)
    string_result = ["0-"] * (total_beats * max_granularity)
    note_on_dims = []
    for i, track in enumerate(mid.tracks):
            total_ticks = 0
            #print('Track {}: {}'.format(i, track.name))
            note_on_in_track = False
            internal_token = ""
            total_ticks = 0
            for msg in track:
                if msg.type == "note_on" :
                    note_on_in_track = True
                    if msg.velocity > 0 :
                        #print(total_ticks * max_granularity / mid.ticks_per_beat)
                        curr_index = round(total_ticks * max_granularity / mid.ticks_per_beat)
                        #print(curr_index)
                        if string_result[curr_index] == "0-" :
                            # if the string at current token is empty
                            internal_token = str(msg.note) + "-"
                        else:
                            # can we sort them as we add them? #that'd be nice!
                            internal_token = string_result[curr_index] + str(msg.note) + "-"

                        string_result[curr_index] = internal_token
                    #increament the total ticks regardless there was a played note or not (velocity>0)
                    total_ticks = total_ticks + msg.time               
    return " ".join(string_result)

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

### params 
### series - is a numpy array representing the song, or the ouput of a net, or the convert_midi_to_time_series method
### midi_program - is the numerical value that corresponds to the desired instument, look at https://soundprogramming.net/file-formats/general-midi-instrument-list/
### output_path - where in file system to write the resulting midi
def write_midi_from_series(series, midi_program, output_path):
    outfile = MidiFile()

    step_size = int(outfile.ticks_per_beat / 8)
    
    track = MidiTrack()
    outfile.tracks.append(track)

    track.append(Message('program_change', program=midi_program))

    delta = 0
    
    for i in range(len(series[0])):
        for j in range(len(series)):
            note = series[j,i]
            if note > 20: # this is to prevent low signals from filtering in
                #print(note)
                track.append(Message('note_on', note=note, velocity=100, time=delta))
                delta = 0

        delta = delta + step_size

    track.append( MetaMessage('end_of_track'))
    
    print("Creating midi file: ", output_path, " from series")
    outfile.save(output_path)

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

### Generates records of size record_size, from the complete timeseries of a midi the target is sparse
def get_sparse_training_data(time_series, record_size):

    nb_classes = 256
    time_series_height = time_series.shape[0]
    time_series_len = time_series.shape[1]

    result_training = np.zeros((time_series_len +1 - record_size, time_series_height, record_size))
    result_labels = np.zeros((time_series_len +1 - record_size , time_series_height * nb_classes))
    #print(result_training.shape)
    idx = 0
    
    #while idx < 10:
    while idx <= time_series_len - record_size -1:
        result_training[idx] = time_series[:,idx:idx+record_size]
        #print(idx)
        #print(result_training[idx])
        internal_label = time_series[:,idx+record_size].astype(int) # shape (5,)
        #print(internal_label)
        targets = internal_label.reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets] # shape (5,256)
        flat_encodings = one_hot_targets.reshape(time_series_height * nb_classes)

        result_labels[idx] = flat_encodings

        idx = idx + 1

    return result_training, result_labels