
import torch
from scipy.special import softmax
import numpy as np




### params
### time_series - original midi as time series
### net -the neural network to use for imagining
### windows - a python list of the form of ["120:160", "200:280", "400:450"] containing ranges to imagine
### returns the time series, with imagined sections
def imagine_midi(time_series, net, windows) :

    series_len = len(time_series[0])
    #print(series_len)
    
    for win in windows:
        tokens = win.split(":")
        win_start = int(tokens[0])
        win_end =  int(tokens[1])
        for i in range(win_start, win_end):
            
            numpy_tensor = time_series[:,i-64:i].astype("float32")
            #print("in:", numpy_tensor)
            tensor_in = torch.unsqueeze(torch.from_numpy(numpy_tensor),0)
            #print(tensor_in)
            pred = net(tensor_in)
            numpy_pred = pred.detach().numpy()
            
            print(numpy_pred) 

            numpy_pred = numpy_pred.reshape(len(time_series), 256)
            
            
            numpy_pred = softmax(numpy_pred, axis=1)
            
            numpy_pred = np.argmax(numpy_pred, axis=1)

               

            time_series[:,i] = numpy_pred

    return time_series.astype('int')