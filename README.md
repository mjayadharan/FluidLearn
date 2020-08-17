# [FluidLearn](https://github.com/mjayadharan/FluidLearn)
-------------------------

FluidLearn is a software package with python interface, capable of solving non-linear fluid flow problems using supervised deep learning techniques. The solution function and the PDE operator are approximated as neural networks, which will be trained using labelled data.  

Conceptually, this API could be used to solve any well-posed PDE system on complex geometric structures, given enough labelled data in the form of boundary and initial conditions. The architecture could also be used for physical parameter estimation and surrogate modelling. As of now, the package is oriented towards PDE systems governing fluid flow problems with many popular flow systems inbuilt.  Users have the option to train the model from external data, visualize the training curves, save the model, reload the model, continue training the saved model or make predictions from the saved models.   

The package could be seen as an application of the [Physics Informed Neural Networks (PINNs)](https://arxiv.org/abs/1711.10561) which are artificial neural nets training with PDE constraints. The idea was first introduced in [this publication](https://arxiv.org/pdf/1711.10561.pdf) in 2017. For more details on the mathematical theory behind PINNs, please visit the website maintained by the authors of the aforementioned publication [here](https://maziarraissi.github.io/PINNs/).  
A graphical representation of a feed forward type neural net used in the training is shown below.



![flow_learn_diagram](https://user-images.githubusercontent.com/35903705/90431457-b2ebd800-e08e-11ea-9bdd-dde98b2673f7.jpg)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__Approximation of PDE operator using a feedforward neural network__

The FluidLearn api is built on top of tensorflow with keras model subclassing. Most of the  details are hidden from the end user, who will be dealing only with the fluidlearn package interface. For developers, with knowledge of keras and tensor flow APIs, who would like more control over the package or would like to add more features could do so easily by inspecting the modulular structure of the package. For all users, except developers, installation of the package from python's official [PyPi distribution](https://pypi.org/project/fluidlearn/) or pip is recommended. The latter users could use the code directly from [here](https://github.com/mjayadharan/FluidLearn/tree/master/fluidlearn) after setting up dependencies.  
While the users will find no problem accessing the package through a regular python script, just like with any other machine learning library, it will be visually advantageous to use a notebook setting like jupyter notebook. For this reason, all the demo examples are available in both python(.py) and jupyter notebook (.ipynb) formats.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![fluidlearn-dependency](https://user-images.githubusercontent.com/35903705/90439301-f5b3ad00-e09a-11ea-87bd-74a873bcfa3f.png)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__FluidLearn package dependency tree__

### Author 
------------

Manu Jayadharan, Department of Mathematics at University of Pittsburgh, 2020.  
email: [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com), [manu.jayadharan@pitt.edu](mailto:manu.jayadharan@pitt.edu)  
[researchgate](https://www.researchgate.net/profile/Manu_Jayadharan)  
[linkedin](https://www.linkedin.com/in/manu-jayadharan/)

## Installation
-----------------------

FluidLearn depends primarily on tensorflow (>=v2.2) and numpy. Make sure you have these packages already available, otherwise please follow the instructions below to install them. Installing all packages inside a separate environment is always recommended in order to prevent version conflicts. You could either use [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) package or a  package manager like [conda](https://docs.anaconda.com/anaconda/install/) to accomplish this.  

#### Installing tensorflow
Installing latest version of tensorflow would automatically install numpy as well. 

#to make sure latest verson of pip is installed.   
`pip install --upgrade pip`    
#installing latest version of  tensorflow.       
`pip install tensorflow`  

Once installed make sure that you have a compatible version of tensorflow by running the following commands inside a py script or notebook.  
`import tensorflow as tf`    
`import numpy as np`  
`tf.__version__ >= '2.2.0'` 


#### Installing FluidLearn

`pip install fluidlearn`

### Other recommended packages for easy visualization

- jupyter notebook for more interactive interface  
    using pip:  
    `pip install notebook`   
    using conda:  
    `conda install jupyter`    
- matplotlib for visualization  
    using pip:  
    `python -m pip install -U pip`  
    `python -m pip install -U matplotlib`  
    or using conda:   
    `conda install matplotlib`

## Getting started with FluidLearn

- Go through [examples](https://github.com/mjayadharan/FluidLearn/tree/master/examples) to understand the user interface of fluidlearn. 
- Examples can be treated as tutorials with the two digit numerals at the beginning of the name indicating the order. For example examples/01_difussion_example is the first example in the series. This example shows how to upload data from a csv file, select nerual architechure, train the model, make prediction and finally how to save and reload the model. 
- All examples are given in both .ipynb and .py formats.
- If you are using the notebook file (.ipynb), make sure that jupyter notebook is installed in the same environment containing fluidlearn and you have started the notebook using `jupyter notebook` in the appropriate environment.  

## Coming in future versions

- More examples demonstrating the abilities of the package.
- More types of nueral network like convolutional nets.
- Building user interface for physical parameter estimation.
