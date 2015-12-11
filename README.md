# CS-281-final

All code requires Berkeley's Caffe library with its Python bindings and all its dependencies.  To install Caffe, go here:  http://caffe.berkeleyvision.org/installation.html  It also requires the Python's PIL library, lmdb library, and Google protobuf library, all of which can be installed via pip.

To train the neural net, unpack the splitwav folder stored [here](https://www.dropbox.com/s/ih10cgozy9pb15b/splitwavsmall.zip?dl=0) into the /data/piano folder.  It contains the training and test sets.  Next, run the build_lmdb.py script in /data/piano to generate the LMDB databases that caffe needs with:

    python build_lmdb.py
    
Next, generate the temporary and output directories by running in the root folder:

    mkdir tmp output
    
Finally, actually train the net by running in the /src directory:
  
    caffe train --solver=solver.prototxt
    
When the net is done training, it will create a file named '.iter_10000.caffemodel' that contains the model.  To actually run our DeepDreams, run in /src:

    python custom_deepdreams_sound.py
    
This will run DeepDreams on a file in splitwav and store the result in /src/out.wav
