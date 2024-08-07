# Meta-Learning-Model-for-Image-Registration
Code published in context of MIDL 2022

## Toolbox
This is a minimal version of our unFAIR toolbox which is intended to close the gap between conventional gradient-based
 optimization methods and neural networks in the context of image registration. It is a Python and Pytorch-based
 framework in development.
 
## Requirements
To use our toolbox, you need Python 3.5 or newer.
The following packages are needed (we make use of a docker environment. The package versions are indicated in the
brackets):
- pytorch       (1.3.1)
- pillow        (9.4.0)
- dill          (0.2.9)
- h5py          (2.9.0)
- numpy         (1.16.3)
- pandas        (0.24.2)
- torchvision   (0.4.2)
- scipy         (1.2.1)
- matplotlib    (3.2.1)

## Data
Images for on-the-fly data generation are provided in images.zip for all data sets mentioned in the paper. Testing
requires HDF file with ground-truth deformations. Sample data to test the full pipeline can be found in minimal_example.zip.
 
    
## Documentation
There is no documentation available yet. This version contains reduced visualization and tracking options. Commonly
used methods such as Tensorboard are provided in the full version.

NOTE: Minimal version not extensively tested. In case of problems please get in touch with the author.

## Author
Frederic Kanter, Institute of Mathematics and Image Computing, University of Lübeck, Germany
frederic.kanter@mic.uni-luebeck.de
