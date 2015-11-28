import wave
import struct
import numpy as np
import os
import caffe
import lmdb


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
    count += 1

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = data.nbytes * 10

env = lmdb.open('../piano_lmdb', map_size=map_size)
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(len(data)):
        datum = caffe.io.array_to_datum(data[i])
        datum.label = labels[i]
        str_id = '{:08}'.format(i)
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())