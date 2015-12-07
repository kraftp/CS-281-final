import numpy as np
import wave
from scipy.io.wavfile import write as wavwrite
import struct
import random

LENWAV = 20000 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4

waveFile = wave.open('../data/pitbull.wav', 'rb')

params = waveFile.getparams()
waveFile.close()

waveFile = wave.open('out.wav', 'wb')
waveFile.setparams(params)
waveFile.setnframes(0)
waveFile.setframerate(waveFile.getframerate() / SAMPLEFACTOR)

for datum in range(0, LENWAV * SAMPLEFACTOR):
    waveFile.writeframes(struct.pack("<h", min(32767, max(-32768, np.random.normal(loc=0, scale=16000)))))
waveFile.close()
