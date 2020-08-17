#!/usr/bin/env python
# Generated from a .ipynb file
# __Author: Manu Jayadharan, University of Pittsburgh, 2020__

# # Using fluidlearn to solve diffusion equation

# Equation to solve: $u_t-\Delta u -f  = 0$
# over domain $\Omega$ from time T_initial to T_final.

# For demonstration purposes we take $f=sin(x_1 + x_2) + tsin(x_1 + x_2)$ and $\Omega = [-2,2]\times [0,1]$ and the time interval to be $[0,1]$, so we can compare the results with the actual solution $u=tsin(x_1 + x_2)$.

#Import fluidlearn package and classes
import fluidlearn
from fluidlearn import dataprocess


# ### Defining the domain and time interval for which the PDE needs to be solved.
# This matters only for generating collocation points and if the user is feeding their own collocation points,
# they can skip this step.

#domain range
X_1_domain = [-2, 2]
X_2_domain = [0, 1]
#time range
T_initial = 0
T_final = 1
T_domain = [T_initial, T_final]

#domain of the problem
domain_bounds = [X_1_domain, X_2_domain, T_domain]


# ### Loading data from a csv file
# - We use the manufactured data with $u=tsin(x_1 + x_2)$ saved in a csv file.
# - Data is saved in the format: ($x_1 , x_2, t, u(x_1, x_2, t)$) as four columns.
# - You could load either preprocess your data to be in this format or load your data
#   from a csv file with similar format.


path_to_data = "data_manufactured/t_sin_x_plus_y.csv"
X_data, Y_data = dataprocess.imp_from_csv(path_to_csv_file=path_to_data,
                                           x_y_combined=True, y_dim=1)


# ### Defining the rhs function $f=sin(x_1 + x_2) + tsin(x_1 + x_2)$ of the PDE.
# We use tensorflow.sin function instead of python functions, we could used numpy.sin as well.

def rhs_function (args, time_dep=True):
        import tensorflow as tf
        if time_dep:
            space_inputs = args[:-1]
            time_inputs = args[-1]
        else:
            space_inputs = args
        
        return tf.sin(space_inputs[0]+space_inputs[1]) + 2*time_inputs*tf.sin(space_inputs[0]+space_inputs[1])


# ### Defining the model architecture

model_type = 'forward'
space_dim = 2 #dimension of Omega
time_depedent_problem = True
n_hid_lay=3 #numberof hidden layers in the neural network
n_hid_nrn=20 #number of neurons in each hidden layer
act_func='tanh' #activation function used for hidden layers:  could be elu, relu, sigmoid
loss_list='mse' #type of error function used for cost functin, we use mean squared error.
optimizer='adam' #type of optimizer for cost function minimization
dom_bounds=domain_bounds #domain bounds where collocation points has to be generated

distribution = 'uniform' #type of distribution used for generating the pde collocation points.
number_of_collocation_points = 5000

batch_size = 32 #batch size for stochastic batch gradient type optimization
num_epochs = 10 #number of epochs used for trainng  


# ### Defining the fluidlearn solver 

diffusion_model = fluidlearn.Solver()

diffusion_model(model_type=model_type,
            space_dim=space_dim,
            time_dep=time_depedent_problem,
            output_dim=1,
            n_hid_lay=n_hid_lay,
            n_hid_nrn=n_hid_lay,
            act_func=act_func,
            rhs_func=rhs_function,
            loss_list=loss_list,
            optimizer=optimizer,
            dom_bounds=dom_bounds,
            load_model=False,
            model_path=None,)


# ### Fitting the model

diffusion_model.fit(
    x=X_data,
    y=Y_data,
    colloc_points=number_of_collocation_points,
    dist=distribution,
    batch_size=batch_size,
    epochs=num_epochs,
)


# ### Resuming Training  the model again for 50 more epochs

diffusion_model.fit(
    x=X_data,
    y=Y_data,
    colloc_points=number_of_collocation_points,
    dist=distribution,
    batch_size=batch_size,
    epochs=50,
)


# ### Demo Using the trained model for predicton
#taking two points from the domain for time t=0.3 and t=0.76 respectively
x_test_points = [[-0.5,0.1,0.3],
                [0.66,0.6,0.76]]
#Predicting the value
y_predicted = diffusion_model.predict(x_test_points)

#finding the true y value for comparing
import numpy as np
x_test_points = np.array(x_test_points)
y_true = np.sin(x_test_points[:,0:1] + x_test_points[:,1:2]) * x_test_points[:,2:3]

#looking at predicted and true solution side by side.
np.concatenate([y_predicted, y_true], axis=1)


# Note that we need more training for further improving the accuracy.

# ### Saving the model to a specified location.
path_to_save_model = "saved_model/model_name"
diffusion_model.save_model(path_to_save_model)


# ### Loading the saved model 
path_to_load_model = "saved_model/model_name"


loaded_diffusion_model = fluidlearn.Solver()

loaded_diffusion_model(space_dim=2,
    time_dep=True,
    load_model=True,
    model_path=path_to_load_model)


# ### Predicting using loaded model
y_predicted = loaded_diffusion_model.predict(X_data)


print("first 10 values of y_predicted: " y_predicted[:10])
