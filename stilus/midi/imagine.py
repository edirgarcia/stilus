
import torch

### params
### pred - prediction output tensor
### net - the network to get the std, and mean from
### returns the net output remapped to ints that correspond to notes
def std_tensor_to_int(pred, net):
    return ((pred * net.midi_dataset.std) + net.midi_dataset.mean).astype(int)


### params
### input - tensor of ints that represents the notes
### net - the network to get the std, and mean from
### returns the input tensor properly converted to input to neural network
def int_to_std_tensor(input, net):
    return ((input - net.midi_dataset.mean) / net.midi_dataset.std)

### params
### time_series - original midi as time series
### net -the neural network to use for imagining
### windows - a python list of the form of ["120:160", "200:280", "400:450"] containing ranges to imagine
### returns the time series, with imagined sections
def imagine_midi(time_series, net, windows) :
    std_time_series = int_to_std_tensor(time_series, net)
    
    series_len = len(std_time_series[0])
    #print(series_len)
    
    for win in windows:
        tokens = win.split(":")
        win_start = int(tokens[0])
        win_end =  int(tokens[1])
        for i in range(win_start, win_end):
            
            numpy_tensor = std_time_series[:,i-64:i].astype("float32")
            #print("in:", numpy_tensor)
            tensor_in = torch.unsqueeze(torch.from_numpy(numpy_tensor),0)
            pred = net(tensor_in)
            numpy_pred = pred.detach().numpy()
            #print("out:", numpy_pred)
            #print("orig:", std_time_series[:,i])
            #std_time_series = np.concatenate((std_time_series,numpy_pred), axis=1)
            std_time_series[:,i] = numpy_pred

    return std_tensor_to_int(std_time_series, net)