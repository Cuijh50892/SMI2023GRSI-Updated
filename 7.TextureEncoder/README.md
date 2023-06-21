## Dependencies
Requirements:
- Python 2.7 or Python 3.6
- PyTorch

The code has been tested with Python 3.6, PyTorch 1.4.0, CUDA 9.2 on Windows10.

## Installation
Download the pre-trained model from https://drive.google.com/file/d/1VMCNe1E8eoNkjsRcvj6bNK23mrU-hD3n/view?usp=sharing

Put the downloaded file in model/

Put 256×256 images in data/train or data/test



### Usage
To train a texture AE:

    run train_256.sh

To encode a texture to latent code:

    run test_256.sh
