import wave
import struct
import numpy as np
import os
import caffe
import matplotlib.pyplot as plt


numwav = len([f for f in os.listdir('splitwav') if '.wav' in f]) #must match batch_size in neural net prototxt

LENWAV = 10000 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4
DIR = 'splitwav'


data = np.zeros((numwav, 1, LENWAV, 1), dtype=int)
labels = np.zeros(numwav, dtype=int)
count = 0

for newFile in np.random.permutation(os.listdir(DIR)):
    if '.wav' in newFile:
        print newFile
        if 'class' in newFile:
            labels[count] = 1
        waveFile = wave.open(DIR + '/' + newFile, 'rb')
        for i in range(0, LENWAV * SAMPLEFACTOR):
            waveData = waveFile.readframes(1)
            if i % SAMPLEFACTOR == 0:
                sound = struct.unpack("<h", waveData)
                data[count,0,i/SAMPLEFACTOR,0] = sound[0]
        print len(data[count,0,:,0])
        plt.scatter(range(0, 10000), data[count,0,:,0])
        plt.show()
    count += 1
