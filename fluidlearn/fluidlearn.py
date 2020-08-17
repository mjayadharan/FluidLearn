"""
Module containng the main classes for PDE solving and parameter estimation using fluidlearn 
"""
try:     
    import numpy
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    
    
    from fluidlearn import fluidmodels
    from fluidlearn import dataprocess
    from fluidlearn import losses
except Exception as e:
        print("Error occured while importing dependency packages. More details:\n",e)
        
__author__ = "Manu Jayadharan"
__copyright__ = "Copyright 2020, fluidlearn"
__credits__ = ["Manu Jayadharan"]
__license__ = ""
__version__ = "0.2.0"
__maintainer__ = "Manu Jayadharan"
__email__ = "manu.jayadharan@pitt.edu"
__status__ = "Development"     

class Solver:
    """
    Main class for solving the PDES using the fluidlearn architechture. Users deal with it, 
    talk about __call__ function
    """
    def __init__(self):
        """
        Doc string goes here
        """
        #Add a block for checking version compabilities.
        
        #class specific variables
        self._time_dep = False
        self._models_dict = {"forward":fluidmodels.ForwardModel}
        self._losses_dict ={"mse":[losses.u_loss, losses.pde_loss]}
        self._dom_bounds =[[]]
        self._trained = False
        
        #place holder for class specific objects from external classes
        self._model = None
        self._model_history = None
        self._data_handler = None

        
    def create_model(self, model_type, space_dim, time_dep,
                      output_dim, n_hid_lay, n_hid_nrn, act_func,
                       rhs_func):
        """
        Doc string goes here
        """
        
        assert (model_type in self._models_dict), "model_type not compatible, " \
        "please try one of the following:\n{}".format(self._models_dict)
        
        #Defining the _model
        self._model = self._models_dict[model_type] (space_dim=space_dim, time_dep=time_dep,
                                                      output_dim=output_dim, n_hid_lay=n_hid_lay,
                                                       n_hid_nrn=n_hid_nrn, act_func = act_func,
                                                        rhs_func = rhs_func)
    
    def compile_model(self, optimizer, loss_list, run_eagerly=False):
        """
        Doc string goes here
        """
        
        assert (self._model !=None), "Model not created here, " \
                                    "run create_model() before compiling:\n"
                                    
        assert (loss_list in self._losses_dict), "given loss functin incompatible. "\
                                                "please give one of {}".format(self._losses_dict)
        self._model.compile(loss=self._losses_dict[loss_list], optimizer=optimizer, run_eagerly=run_eagerly)
    
    def get_model_summary(self):
        """
        Doc string goes her
        """
        
        if self._model != None:
            self._model.summary()
        else:
            raise Exception("The _model hasn't been built yet, _model is None as of now.")
        
    
    def fit(self, x, y, x_val=[], y_val=[], validation=False, colloc_points=100, dist="uniform",
            batch_size=32, epochs=1,):
        """
        Doc string goes here.
        """
        assert (self._model !=None and self._data_handler != None), "Model and data handler" \
                                    " not created yet, run create_model() before compiling:\n"                        
        assert (len(self._dom_bounds[0])>0), "Problem domain bounds missing ." \
                                    " dom_bounds required for generating collocation points"
        
        #preparing data                                    
        X_tr, Y_tr = self._data_handler.get_training_data(X_data=x, Y_data=y,
                                                           X_col_points=colloc_points, dist=dist)
        
        #fitting model
        self._model_history = self._model.fit(x=X_tr, y=[Y_tr,Y_tr], batch_size=batch_size, epochs=epochs)
        
        self._trained = True
        
    def predict(self, x):
        """
        Doc string goes here
        """
        assert (self._trained), "Model is not trained yet, please call fit method for training"
        
        x_prepared = self._data_handler.prepare_input_data(x)
        return self._model.predict(x=x_prepared)
    
    def evaluate(self, x, y):
        """
        Doc string goes here
        """
        assert (self._trained), "Model is not trained yet, please call fit method for training"
        #preparing data 
        X_prepared, y_prepared = self._data_handler.prepare_input_data(X_data=x, Y_data=y)
        #evaluating model
        return self._model.evaluate(x=X_prepared, y=y_prepared)
        
    def save_model(self, path_to_save):
        """
        Doc string goes here
        """
        assert (self._trained), "Model is not trained yet, please call fit method for training"
        
        try:
            assert(type(path_to_save) == str), "path_to_save has to be a string object"
            self._model.save(path_to_save)
        except Exception as e:
            raise Exception("Error occured while saving model of type {}, as follows: \n{}".format(type(e)),e)
     
    def load_model(self, path_to_saved_model):
        """
        Doc string goes here
        """
        self._model = keras.models.load_model(path_to_saved_model,
                                              custom_objects={"u_loss": losses.u_loss,
                                                             "pde_loss": losses.pde_loss})
        
           
    def __call__(self, model_type="forward", space_dim=1, time_dep=False,
                      output_dim=1, n_hid_lay=3, n_hid_nrn=20, act_func = "tanh",
                       rhs_func = None, loss_list="mse", optimizer="adam",
                       dom_bounds=[[]],
                       load_model=False, model_path=None):
        """
        Doc string goes here
        """
        self._time_dep = time_dep
        self._data_handler = dataprocess.DataPreprocess(space_dim=space_dim, dom_bounds=dom_bounds, time_dep=time_dep)
        self._dom_bounds = dom_bounds
        
        if not load_model:
            self.create_model(model_type=model_type, space_dim=space_dim, time_dep=time_dep,
                          output_dim=output_dim, n_hid_lay=n_hid_lay, n_hid_nrn=n_hid_nrn, act_func = act_func,
                           rhs_func = rhs_func)
            
            #compiling the _model for the first time
            self.compile_model(optimizer=optimizer, loss_list=loss_list)
        elif load_model:
            assert (model_path), "model_path empty: need path inorder to load a previously saved _model"
            
            self.load_model(model_path)
            self._trained = True
    
        
# if __name__ == "__main__":
#     
#     #domain for the problem
#     #domain range
#     X_1_domain = [-2, 2]
#     X_2_domain = [0, 1]
#     #time range
#     T_initial = 0
#     T_final = 1
#     T_domain = [T_initial, T_final]
#     domain_bounds=[X_1_domain,X_2_domain,T_domain]
#     
# #     #Defining the class and fitting a model
# #     my_solver = Solver()
# #     my_solver(model_type="forward", space_dim=2, time_dep=True,
# #                       output_dim=1, n_hid_lay=3, n_hid_nrn=20, act_func = "tanh",
# #                        rhs_func = None, loss_list="mse", optimizer="adam",
# #                        dom_bounds=domain_bounds,
# #                        load_model=False, model_path=None)
# #     print("reached here")
# #     X_data, Y_data = dataprocess.imp_from_csv("data_manufactured/t_sin_x_plus_y.csv",
# #                                              True,1)
# #     my_solver.fit(x=X_data, y=Y_data, colloc_points=1000, epochs=5)
# #     my_solver.save_model("new_saved_model/")
# 
#     #loading a model
#     my_solver = Solver()
#     my_solver(load_model=True, model_path="new_saved_model/")
#     #Next thing to do: add predict method make sure it works. Using thta make sure taht loaded model is working.
    
        
