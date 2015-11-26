import wave
import struct
import numpy as np
import os
import caffe


numwav = len([f for f in os.listdir('splitwav') if '.wav' in f]) #must match batch_size in neural net prototxt

LENWAV = 10000 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4
DIR = 'splitwav'


data = np.zeros((numwav, 1, LENWAV, 1), dtype=np.float32)
labels = np.zeros(numwav, dtype=np.float32)
count = 0

for newFile in os.listdir(DIR):
    if '.wav' in newFile:
        print newFile
        if 'class' in newFile:
            labels[count] = 1
        waveFile = wave.open(DIR + '/' + newFile, 'rb')
        for i in range(0, LENWAV * SAMPLEFACTOR):
            waveData = waveFile.readframes(1)
            if i % SAMPLEFACTOR == 0:
                sound = struct.unpack("<h", waveData)
                data[count,0,i/SAMPLEFACTOR,0] = int(sound[0])
    count += 1

solver = caffe.get_solver('solver.prototxt')
net = solver.net
net.set_input_arrays(data, labels) # change memory_data_param.width in simple.prototxt to 2 if you want to use data_fft
net.backward()
net.forward()
