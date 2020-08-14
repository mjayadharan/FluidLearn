# FluidLearn
Software package to solve PDEs governing fluid flow using Physics Inspired Neural Networks (PINNs) and estimate physical parameters using the trained network.
Uses Keras API with TensorFlow2 as the backend. Developed as part of a project to study the use of artificial neural networks in solving computational fluid dynamics problem.   


### Note: 
- As of now, the simulators are embdedded inside jupyter notebooks.    
- Matplotlib is used as a visualization tool along with TensorBoard.  
- Only feed forwards networks are implemented so far. Expect more advanced network structures to be implemented in future.  
- Python module version of the package will be uploaded upon developing the UI  of the package to certain satisfactory level.   

## Author
-----------
Manu Jayadharan, Department of Mathematics at University of Pittsburgh, 2020

email: [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com), [manu.jayadharan@pitt.edu](mailto:manu.jayadharan@pitt.edu)  
[reserachgate](https://www.researchgate.net/profile/Manu_Jayadharan)  
[linkedin](https://www.linkedin.com/in/manu-jayadharan/)

## Installing python packages
----------------------
- You could install the required packages manually following the instruction below or run a shell script (inside the terminal) which will execute the steps below one by one. Please contact the author if you would like to get a shell script for installation or need assistance regarding setting up the dependency packages.  
- Easiest way to satisfy packages dependencies is to use [anaconda](https://www.anaconda.com/).  
- The following steps assume that you work on a terminal, if you work on windows, you could install anaconda following the instructions from [here](https://docs.anaconda.com/anaconda/install/windows/) and install the following packages using the windows GUI: numpy, pandas, matplotlib, scikit-learn, tensorflow, jupyter.  
- A good documentation on installatin of anaconda can be found [here.](https://docs.anaconda.com/anaconda/install/linux/)
- Creating an environment in anaconda to keep the package versions consistent:  
`conda create --name myenv`
`conda activate myenv`
- Installing jupyter notebook:  
`conda install jupyter`
- Starting a jupyter notebook:  
 `jupyter notebook`  
__Installing required packages:__   
`conda install numpy matplotlib tensorflow`  
__Installing recommended packages:__  
`conda install numpy pandas matplotlib scikit-learn tensorflow jupyter`  

