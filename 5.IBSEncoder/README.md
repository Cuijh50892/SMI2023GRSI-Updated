## Dependencies
Requirements:
- Python 2.7+ with Numpy, Scipy and Matplotlib

The code has been tested with Python 2.7, TensorFlow 1.3.0, TFLearn 0.3.2, CUDA 8.0 and cuDNN 6.0 on Ubuntu 16.04.


## Installation

To be able to train your own model you need first to _compile_ the EMD/Chamfer losses. In external/structural_losses
```
cd external

with your editor modify the first three lines of the makefile to point to 
your nvcc, cudalib and tensorflow library.

make
```

Store point-clouds in data/uniform_samples_2048/00000000

Use the function snc_category_to_synth_id, defined in src/in_out/, to map a class name such as "train" to its synthetic_id: "0000000".


### Usage
To train a point-cloud AE:

    python train.py

To encode a point-cloud to latent code:

    python encode.py

