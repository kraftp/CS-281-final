import wave
import struct
import numpy as np
import os
import caffe
import lmdb
import shutil

numwav = len([f for f in os.listdir('splitwav') if '.wav' in f]) #must match batch_size in neural net prototxt

LENWAV = 2*40000/4 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4
DIR = 'splitwav'
fft_chunksize = 1000

shutil.rmtree('../piano_fft_train_lmdb', ignore_errors=True)
shutil.rmtree('../piano_fft_test_lmdb', ignore_errors=True)

data = np.zeros((numwav, 1, LENWAV, 2), dtype=float)
labels = np.zeros(numwav, dtype=int)
count = 0
fft = np.zeros(LENWAV, dtype=complex)

np.random.seed(666)

for newFile in np.random.permutation(os.listdir(DIR)):
    assert('.wav' in newFile)
    print newFile
    if 'class' in newFile:
        labels[count] = 1
    waveFile = wave.open(DIR + '/' + newFile, 'rb')
    for i in range(0, LENWAV * SAMPLEFACTOR):
        waveData = waveFile.readframes(1)
        if i % SAMPLEFACTOR == 0:
            sound = struct.unpack("<h", waveData)
            data[count,0,i/SAMPLEFACTOR,0] = sound[0]
    for i in range(0, LENWAV, fft_chunksize):
        fft[i:i+fft_chunksize] = np.fft.fft(data[count,0,i:i+fft_chunksize,0])
    data[count,0,:,0] = np.real(fft)
    data[count,0,:,1] = np.imag(fft)
    #data[count,0,:,0] *= 20000. / np.max(np.abs(data[count,0,:,0]))
    count += 1
# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = data.nbytes * 10

env = lmdb.open('../piano_fft_train_lmdb', map_size=map_size)
env2 = lmdb.open('../piano_fft_test_lmdb', map_size=map_size)
with env.begin(write=True) as txn:
    with env2.begin(write=True) as txn2:
        # txn is a Transaction object
        for i in range(len(data)):
            datum = caffe.io.array_to_datum(data[i])
            datum.label = labels[i]
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            if np.random.randint(1,10) != 1:
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            else:
                txn2.put(str_id.encode('ascii'), datum.SerializeToString())
