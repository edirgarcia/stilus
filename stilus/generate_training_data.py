from midi.utils import convert_midi_to_time_series, get_sparse_training_data, convert_midi_to_string 
from mido import Message, MidiFile, MidiTrack
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
import pandas as pd

# <your_path>\Source\Repos\stilus> python stilus/generate_training_data.py -i midi/training/ -c bach

parser = argparse.ArgumentParser(description="Generate training data from a given folder of midi")
parser.add_argument("-i","--inputRootPath", default="midi/training/", help="The root path for the input", )
parser.add_argument("-c","--composer", default="bach", help="The path to the input from inputRoot, this will be used to name the output data folder")

args = parser.parse_args()

print("Generating training data...")

in_path = args.inputRootPath
composer = args.composer

training_result = None
labels_result = None

for root, dirs, files in os.walk(in_path+composer+"/"):
    for file in files :
        abs_path = os.path.join(root, file)
        print("processing file: ", abs_path)
        mid = MidiFile(abs_path)

        #time_series = convert_midi_to_time_series(mid, 5, 5, 8)
        time_series = convert_midi_to_string(mid, 5, 8)

        training_data, training_labels = get_sparse_training_data(time_series, 64)

        if training_result is None :
            training_result = training_data
            labels_result = training_labels
        else :
            training_result = np.vstack((training_result, training_data))
            labels_result = np.vstack((labels_result, training_labels)) 

print(training_result.shape)
print(labels_result.shape)


train_data, test_data, train_labels, test_labels = train_test_split(training_result, labels_result , test_size=.15, random_state=42, shuffle=True)

out_path = "data/" + composer 
if not os.path.isdir(out_path):
    os.makedirs(out_path)

print("test data resulting tensor shape:", test_data.shape)
np.save(out_path + "/test_data", test_data)

print("test labels resulting tensor shape:", test_labels.shape)
np.save(out_path + "/test_labels", test_labels)

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=.15, random_state=42, shuffle=True)

print("train data resulting tensor shape:", train_data.shape)
np.save(out_path + "/training_data", train_data)

print("train labels resulting tensor shape:", train_labels.shape)
np.save(out_path + "/training_labels", train_labels)

print("val data resulting tensor shape:", val_data.shape)
np.save(out_path + "/validation_data", val_data)

print("val labels resulting tensor shape:", val_labels.shape)
np.save(out_path + "/validation_labels", val_labels)





