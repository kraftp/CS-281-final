import numpy as np
import wave
from scipy.io.wavfile import write as wavwrite
import struct
import random

LENWAV = 20000 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4
img = np.zeros((1, 1, LENWAV, 1), dtype=int)

waveFile = wave.open('../data/pitbull.wav', 'rb')
for i in range(0, LENWAV * SAMPLEFACTOR):
    waveData = waveFile.readframes(1)
    if i % SAMPLEFACTOR == 0:
        sound = struct.unpack("<h", waveData)
        img[0,0,i/SAMPLEFACTOR,0] = random.uniform(-400, 400)

print "NEW SAMPLE RATE", waveFile.getframerate() / SAMPLEFACTOR

wavwrite('../data/noise.wav', waveFile.getframerate() / SAMPLEFACTOR, img.flatten() / float(np.max(np.abs(img.flatten()),axis=0)))
