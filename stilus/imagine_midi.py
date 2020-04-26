import argparse
import sys
import midi.utils as utl
import models as m
import midi.imagine as img

from mido import MidiFile



parser = argparse.ArgumentParser(description="Insert imagined segments into a midi using a pretrained neural network")
parser.add_argument("-i","--inputMidi", required=True, help="The path to the input midi")

parser.add_argument("-t","--networkType", required=True, help="The type of the model to use, good values are ConvNet_1_0_2, and TransformerNet_1_0_2")
parser.add_argument("-n","--networkPath", required=True, help="The path to the pretrained model. This is a .ckpt file")

parser.add_argument("-d","--dataPath", required=True, help="The path to the data that trained the model, to sigure out std and means to map back to original distrib")

parser.add_argument("-r","--ranges", required=True, metavar='range', nargs='+', help="The ranges in timesteps to imagine e.g. \"128:160\", \"200:280\" ")
# metavar='N', type=int, nargs='+'
parser.add_argument("-o","--outputMidi", help="The path to the output midi, if not given it will be assumed from input and model")

args = parser.parse_args()


#read in input midi
file_path = args.inputMidi
tokens = file_path.split("/")
file_name = tokens[len(tokens)-1]
file_name_no_ext = file_name.split(".")[0]

mid = MidiFile(file_path)

timeseries_tensor = utl.convert_midi_to_time_series(mid,5,5,8)

net = None

if args.networkType == "ConvNet_1_0_2":
    net = m.ConvNet_1_0_2().load_from_checkpoint(args.networkPath)
elif args.networkType == "TransformerNet_1_0_2":
    net = m.TransformerNet_1_0_2().load_from_checkpoint(args.networkPath)
else:
    print("Not supported type!")
    sys.exit()

net.eval()

net.set_data_path(args.dataPath)

generated_series = img.imagine_midi(timeseries_tensor, net, args.ranges)

out_path = 'midi/output/' + file_name_no_ext +"_" + net.name  + '.mid'
utl.write_midi_from_series(generated_series, 2, out_path )