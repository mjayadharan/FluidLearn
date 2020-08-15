"""
Module containng the main classes for PDE solving and parameter estimation using fluidlearn 
"""
from sklearn.ensemble._hist_gradient_boosting import loss

try:     
    import numpy
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    
    import fluidmodels
    import dataprocess
    import losses
except Exception as e:
        print("Error occured while importing dependency packages. More details:\n",e)
        
__author__ = "Manu Jayadharan"
__copyright__ = "Copyright 2020, fluidlearn"
__credits__ = ["Manu Jayadharan"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Manu Jayadharan"
__email__ = "manu.jayadharan@pitt.edu"
__status__ = "Development"     

class Solver:
    """
    Main class for solving the PDES using the fluidlearn architechture. Users deal with it, 
    talk about __call__ function
    """
    
    def __init__(self):
        
        self.model = None
        self.models_list = {"forward":fluidmodels.ForwardModel}
        self.loss_list ={"mse":[losses.u_loss, losses.pde_loss]}
        
    def create_model(self, model_type="forward", space_dim=1, time_dep=False,
                      output_dim=1, n_hid_lay=3, n_hid_nrn=20, act_func = "tanh",
                       rhs_func = None):
        """
        Doc string goes here
        """
        
        assert (model_type in self.models_list), "model_type not compatible, " \
        "please try one of the following:\n{}".format(self.models_list)
        
        #Defining the model
        self.model = self.models_list[model_type] (space_dim=1, time_dep=False, output_dim=1,
                 n_hid_lay=3, n_hid_nrn=20, act_func = "tanh", rhs_func = None)
    
    def compile_model(self, optimizer="Adam", loss_list="mse", run_eagerly=False):
        """
        Doc string goes here
        """
        
        assert (self.model !=None), "Model not created here, " \
                                    "run create_model() before compiling:\n"
        self.model.compile(loss_list=self.loss_list[loss], optimizer=optimizer, run_eagerly=False)
    
    def get_model_summary(self):
        """
        Doc string goes her
        """
        
        if self.model != None:
            self.model.summary()
        else:
            raise Exception("The model hasn't been built yet, model is None as of now.")
        
    def load_model(self, path_to_saved_model):
        pass
        
    def __call__(self, model_type="forward", space_dim=1, time_dep=False,
                      output_dim=1, n_hid_lay=3, n_hid_nrn=20, act_func = "tanh",
                       rhs_func = None, loss_list="mse", optimizer="adam",
                       load_model=False, model_path=None):
        """
        Doc string goes here
        """
        if not load_model:
            self.create_model(self, model_type="forward", space_dim=1, time_dep=False,
                          output_dim=1, n_hid_lay=3, n_hid_nrn=20, act_func = "tanh",
                           rhs_func = None)
            
            #compiling the model for the first time
            self.compile_model(optimizer=optimizer, loss_list=loss_list)
        elif load_model:
            assert (model_path), "model_path empty: need path inorder to load a previously saved model"
            #Add codes to load code from a folder here
            pass
        
if __name__ == "__main__":
    my_solver = Solver()
    my_solver.create_model(model_type="forward")


    
        