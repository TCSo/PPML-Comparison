# PPML-Comparison
**Overview**<br/>
This repo contains code to run comparisons among different PPML architectures, including: 

- [HE-MAN-Concrete](https://github.com/smile-ffg/he-man-concrete)
- [ZAMA Concrete ML](https://github.com/zama-ai/concrete-ml)
- [Aby 3](https://github.com/ladnir/aby3) (not implemented)
- [Rossetta](https://github.com/LatticeX-Foundation/Rosetta) (not implemented)

**Usage**<br/>
To get started, clone the repository by running 
```
git clone https://github.com/TCSo/PPML-Comparison.git
```
cd into the PPML-Comparison directory and run 
```
make
```
This will run all of the comparison code, specifically: 
- clean the directory
- training the necessary models using planetext with MNIST training data, and generate the onnx models as well as the testing input and output files
- run the models and data in the HE-MAN-Concrete framework and generate evaluation statistics
- run the models and data in the ZAMA Concrete ML framework and generate evaluation statistics

**Content**<br/>
The content of the repository includes
- *models.py*: contains the pytorch models used for comparison, training, and data generation
- *Makefile*: contains all the make commands for easy usage of this repository
- *notebook.ipynb*: to visualize some data
- *data*: contains all the MNIST data that we will be using to evaluate the PPML encrypted models
- *calibration-data*: contains a subset (representative) set of the MNIST data for calibrating the encrypted models

**Acknowledgement**<br/>
Part of the code (training loop) is referenced from UC Berkeley's CS182/282A assignments
