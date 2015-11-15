# operates on wavs, which come from `lame --decode in.mp3 out.wav`
import numpy as np
import scipy.io.wavfile
rate, sound = scipy.io.wavfile.read('../data/bach1.wav')
nsamples = 10
sample_size = 65536
data = np.empty((nsamples, 1, sample_size, 1), dtype=np.float32) # nsamples, 1, len, 1
data_fft_cpx = np.empty((nsamples, 1, sample_size, 1), dtype=np.complex64)
data_fft = np.empty((nsamples, 1, sample_size, 2), dtype=np.float32)
for i in range(nsamples):
    data[i,0,:,0] = sound[10000 + 4 * sample_size * i:10000 + sample_size * (4 * i + 1),0] # generate tractable samples
    data_fft_cpx[i,0,:,0] = np.fft.fft(data[i,0,:,0])
data_fft[:,:,:,0:1] = np.real(data_fft_cpx)
data_fft[:,:,:,1:2] = np.imag(data_fft_cpx)
del data_fft_cpx
labels = np.ones((nsamples), dtype=np.float32)
import caffe
solver = caffe.get_solver('solver.prototxt')
net = solver.net
net.set_input_arrays(data, labels) # change memory_data_param.width in simple.prototxt to 2 if you want to use data_fft
net.backward()
