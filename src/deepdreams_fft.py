# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import matplotlib.pyplot as plt
import os
import caffe
import wave
import struct
from scipy.io.wavfile import write as wavwrite

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    #img = PIL.Image.fromarray(a)
    # show on each update
    # PIL.Image.fromarray(a).save(f, fmt)
    # Image(data=f.getvalue()).show()

caffe_path = os.path.abspath(os.path.join(os.path.join(os.path.join(caffe.__file__, os.pardir), os.pardir), os.pardir))
model_path = '.'
net_fn   = os.path.join(model_path, 'deploy_fft.prototxt')
param_fn = os.path.join(model_path, '.fft2.caffemodel')
tmp_file = '../tmp/tmp.prototxt'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open(tmp_file, 'w').write(str(model))

# TODO : CHANGE THE MEAN
net = caffe.Classifier(tmp_file, param_fn,
                       mean = np.float32([0]),
                       channel_swap = (0,)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(net, step_size=1.5, end='conv1',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    # HIDDEN BECAUSE PROBABLY NOT RELEVANT TO MUSIC
    # if clip:
    #     bias = net.transformer.mean['data']
    #     src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='conv1', clip=True, **step_params):

    # FYI
    print net.blobs.keys()

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,1,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            # HIDDEN BECAUSE PROBABLY NOT RELEVANT TO MUSIC
            vis = deprocess(net, src.data[0])
            # if not clip: # adjust image contrast if clipping is disabled
            #     vis = vis*(255.0/np.percentile(vis, 99.98))
            # showarray(vis)
            print octave, i, end, vis.shape

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

LENWAV = 20000 #Must be <= 40000 / SAMPLEFACTOR for one-second wav files
SAMPLEFACTOR = 4
img = np.zeros((LENWAV, 1, 1), dtype=int)

waveFile = wave.open('../data/piano/splitwav/other1-1a.wav', 'rb')
for i in range(0, LENWAV * SAMPLEFACTOR):
    waveData = waveFile.readframes(1)
    if i % SAMPLEFACTOR == 0:
        sound = struct.unpack("<h", waveData)
        img[i/SAMPLEFACTOR, 0, 0] = sound[0]

plt.scatter(range(LENWAV), img[:,0,0])
plt.show()

fft = np.zeros(LENWAV, dtype=complex)
data = np.zeros((LENWAV, 2, 1), dtype=float)
fft_chunksize = 1000
for i in range(0, LENWAV, fft_chunksize):
    fft[i:i+fft_chunksize] = np.fft.fft(img[i:i+fft_chunksize,0,0])
data[:,0,0] = np.real(fft)
data[:,1,0] = np.imag(fft)
out = np.zeros(LENWAV, dtype=int)

output = deepdream(net, data, octave_n=4, iter_n=100)
params = waveFile.getparams()
waveFile.close()

waveFile = wave.open('out.wav', 'wb')
waveFile.setparams(params)
waveFile.setnframes(0)
waveFile.setframerate(waveFile.getframerate() / SAMPLEFACTOR)

fft[:] = output[:,0,0] + 1j * output[:,1,0]
for i in range(0, LENWAV, fft_chunksize):
    out[i:i+fft_chunksize] = np.real(np.fft.ifft(fft[i:i+fft_chunksize]))

for datum in out:
    waveFile.writeframes(struct.pack("<h", min(32767, max(-32768, datum))))
waveFile.close()

plt.scatter(range(LENWAV), out)
plt.show()
