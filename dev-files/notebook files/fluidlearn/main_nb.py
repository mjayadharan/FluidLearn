import numpy as np
import tensorflow as tf
from tensorflow import keras

import os

import dataprocess
import fluidmodels
import losses

#python version of jupyter notebook file, just for testing the packages.

# ### Manufacturing data for trainig

# In[2]:


np.random.seed(123)
pde_data_size = 2000
bc_data_size = 400

#domain range
X_1_domain = [-2, 2]
X_2_domain = [0, 1]
#time range
T_initial = 0
T_final = 1
T_domain = [T_initial, T_final]

#space data
space_dim = 2
# X_1_tr_pde = np.random.uniform(X_1_domain[0], X_1_domain[1], pde_data_size).reshape(pde_data_size,1)
# X_2_tr_pde = np.random.uniform(X_2_domain[0], X_2_domain[1], pde_data_size).reshape(pde_data_size,1)
# # X_tr_pde = np.random.uniform(-1,1,pde_data_size*space_dim).reshape(pde_data_size,space_dim)
# 
# #temporal data
# X_tr_time = np.random.uniform(T_initial, T_final, pde_data_size).reshape(pde_data_size,1)
# 
# X_tr_pde = np.concatenate([X_1_tr_pde, X_2_tr_pde, X_tr_time],axis=1)
# X_1_tr_pde.shape
# 
# 
# # In[3]:
# 
# 
# X_tr_pde.shape
# 
# 
# # ### Looking at the scatter plot of data
# 
# # In[4]:
# 
# 
# # ### Defining the labels(true values) for the training data
# 
# # In[5]:
# 
# 
# Y_tr_pde = np.zeros((X_tr_pde.shape[0],1))
# # Y_tr_pde = X_tr_pde[:,0:1]
# 
# 
# # In[6]:
# 
# 
# Y_tr_pde = np.concatenate([Y_tr_pde,np.zeros((Y_tr_pde.shape[0],1))],axis=1)
# Y_tr_pde.shape
# 
# 
# # ## BC data 
# 
# # In[7]:
# 
# 
# # bc_data_size = 100
# 
# X_bc_left = np.random.uniform(X_2_domain[0],X_2_domain[1],
#                               bc_data_size).reshape(bc_data_size,1)
# X_bc_left = np.concatenate([X_1_domain[0]*np.ones((bc_data_size,1)),
#                             X_bc_left], axis=1)
# X_bc_left = np.concatenate([X_bc_left,
#                             np.random.uniform(T_initial, T_final, bc_data_size).reshape(bc_data_size,1)], 
#                             axis=1)
# 
# X_bc_bottom = np.random.uniform(X_1_domain[0],X_1_domain[1],
#                                 bc_data_size).reshape(bc_data_size,1)
# X_bc_bottom = np.concatenate([X_bc_bottom, X_2_domain[0]*np.ones((bc_data_size,1))],
#                              axis=1)
# X_bc_bottom = np.concatenate([X_bc_bottom,
#                               np.random.uniform(T_initial, T_final, bc_data_size).reshape(bc_data_size,1)], axis=1)
# 
# X_bc_right = np.random.uniform(X_2_domain[0],X_2_domain[1],
#                               bc_data_size).reshape(bc_data_size,1)
# X_bc_right = np.concatenate([X_1_domain[1]*np.ones((bc_data_size,1)),
#                             X_bc_right], axis=1)
# X_bc_right = np.concatenate([X_bc_right,
#                             np.random.uniform(T_initial, T_final, bc_data_size).reshape(bc_data_size,1)], 
#                             axis=1)
# 
# X_bc_top = np.random.uniform(X_1_domain[0],X_1_domain[1],
#                                 bc_data_size).reshape(bc_data_size,1)
# X_bc_top = np.concatenate([X_bc_top, X_2_domain[1]*np.ones((bc_data_size,1))],
#                           axis=1)
# X_bc_top = np.concatenate([X_bc_top,
#                             np.random.uniform(T_initial, T_final, bc_data_size).reshape(bc_data_size,1)], 
#                             axis=1)
# 
# X_bc = np.concatenate([X_bc_left, X_bc_bottom, X_bc_right, X_bc_top],axis=0)
# 
# #Add iniital condition below: add them to be X_ic and finallly concatenate bc and ic to get X_bc_ic
# X_ic = np.random.uniform(X_1_domain[0],X_1_domain[1],
#                                 bc_data_size).reshape(bc_data_size,1)
# X_ic = np.concatenate([X_ic, np.random.uniform(X_2_domain[0],X_2_domain[1],
#                               bc_data_size).reshape(bc_data_size,1)],
#                           axis=1)
# X_ic = np.concatenate([X_ic,
#                             T_initial*np.ones((bc_data_size,1))], 
#                             axis=1)
# 
# 
# # In[8]:
# 
# 
# X_bc_ic = np.concatenate([X_bc, X_ic],axis=0)
# 
# 
# # In[9]:
# 
# 
# X_bc_ic
# 
# 
# # In[10]:
# 
# 
# Y_bc_ic = np.sin(X_bc_ic[:,0:1] + X_bc_ic[:,1:2]) * X_bc_ic[:,2:3]
# # Y_bc = np.concatenate([Y_bc, np.ones((Y_bc.shape[0],1))], axis=1 )
# 
# 
# # ### Saving the data to a csv file
# 
# # In[11]:
# 
# 
# combined_data = np.concatenate([X_bc_ic,Y_bc_ic],axis=1)
# 
# 
# # In[12]:
# 
# 
# dataprocess.save_to_csv(combined_data, "data_manufactured/t_sin_x_plus_y")


# In[13]:


X_data, Y_data = dataprocess.imp_from_csv("data_manufactured/t_sin_x_plus_y.csv",
                                             True,1)


# In[14]:


data_processor = dataprocess.DataPreprocess(2, dom_bounds=[X_1_domain,X_2_domain,T_domain], time_dep=True)


# In[15]:


X_tr, Y_tr = data_processor.get_training_data(X_data,Y_data, X_col_points=pde_data_size)


# In[16]:


# X_tr = np.concatenate((X_tr_pde, X_bc), axis=0)
# Y_tr = np.concatenate((Y_tr_pde, Y_bc), axis=0)


# ## Training the model


# In[18]:


def rhs_function (args, time_dep=True):
        if time_dep:
            space_inputs = args[:-1]
            time_inputs = args[-1]
        else:
            space_inputs = args
        
        return tf.sin(space_inputs[0]+space_inputs[1]) + 2*time_inputs*tf.sin(space_inputs[0]+space_inputs[1])


# In[19]:


pinn_model = fluidmodels.ForwardModel(space_dim=2, time_dep=True, rhs_func = None)
# pinn_model = FluidModels.ForwardModel(space_dim=2, time_dep=True, rhs_func = None)


# In[20]:


# #Loss coming from the boundary terms
# def u_loss(y_true, y_pred):
# #     print("\n\nreached here 1 \n\n\n")
#     y_true_act = y_true[:,:-1]
#     at_boundary = tf.cast(y_true[:,-1:,],bool)
#     u_sq_error = (1/2)*tf.square(y_true_act-y_pred)
# #     print("\n\nreached here 2 \n\n\n")
# #     print("\nu_loss: ",tf.where(at_boundary, u_sq_error, 0.))
#     return tf.where(at_boundary, u_sq_error, 0.)

# #Loss coming from the PDE constrain
# def pde_loss(y_true, y_pred):
#     y_true_act = y_true[:,:-1]
#     at_boundary = tf.cast(y_true[:,-1:,],bool)
#     #need to change this to just tf.square(y_pred) after pde constrain is added to grad_layer
#     pde_sq_error = (1/2)*tf.square(y_pred)
# #     print("\npde_loss: ",tf.where(at_boundary,0.,pde_sq_error))
#     return tf.where(at_boundary,0.,pde_sq_error)


# In[21]:


# loss_1 = Losses.u_loss
# loss_2 = Losses.pde_loss


# In[22]:


pinn_model.compile(loss=[losses.u_loss, losses.pde_loss], optimizer="adam")
# pinn_model.compile(loss=u_loss, optimizer=keras.optimizers.SGD(lr=1e-3))


# In[23]:


# pinn_model.fit(x=[X_tr[:,0:1], X_tr[:,1:2], X_tr[:,2:3]], y=[Y_tr, Y_tr], epochs=10)
history = pinn_model.fit(x=X_tr, y=[Y_tr, Y_tr], epochs=10)


# In[32]:
# 
# 
# 
# 
# 
# # In[27]:
# 
# 
# # pinn_model.compile(loss=[u_loss,pde_loss], optimizer=keras.optimizers.SGD(lr=1e-4))
# # pinn_model.fit(x=[X_tr[:,0:1], X_tr[:,1:2]], y=[Y_tr, Y_tr], epochs=10)
# 
# 
# # In[28]:
# 
# 
# pinn_model.summary()


# ### Testing the model

# In[29]:


# X_test_st = np.random.uniform(-0.5,0.5,20*dim_d).reshape(20,dim_d)


# In[30]:

# 
# #space test data
# test_dat_size = 100
# X_test_st = np.random.uniform(X_1_domain[0],X_1_domain[1],test_dat_size).reshape(test_dat_size,1)
# X_test_st = np.concatenate([X_test_st, np.random.uniform(X_2_domain[0],X_2_domain[1],test_dat_size).reshape(test_dat_size,1)], axis=1)
# #temporal test data
# X_test_time = np.random.uniform(T_initial,T_final,test_dat_size).reshape(test_dat_size,1)
# 
# X_test_st = np.concatenate([X_test_st, X_test_time],axis=1)
# 
# 
# # In[31]:
# 
# 
# X_test_st = data_processor.prepare_input_data(X_test_st)
# 
# 
# # In[32]:
# 
# 
# # Y_test = pinn_model.predict(x=[X_test_st[:,0:1], X_test_st[:,1:2], X_test_st[:,2:3]])
# # pinn_model.predict(x=[X_tr[0:40,0:1], X_tr[0:40,1:2], X_tr[0:40,2:3]]) [0]
# Y_test = pinn_model.predict(x=X_test_st)
# len(Y_test)
# 
# 
# # In[33]:
# 
# 
# Y_test_true = np.sin(X_test_st[0] + X_test_st[1]) * X_test_st[2]
# Y_eval = np.concatenate([Y_test_true,np.ones((Y_test_true.shape[0],1))], axis=1)
# 
# 
# # In[34]:
# 
# 
# # pinn_model.evaluate(x=[X_test_st[:,0:1], X_test_st[:,1:2]], y= Y_eval)
# 
# 
# # In[35]:
# 
# 
# np.concatenate([Y_test_true, Y_test], axis=1)
# 
# 
# # In[36]:
# 
# 
# plt.scatter(Y_test_true,Y_test)
# plt.title("true vs predicted solution")
# plt.xlabel("True solution")
# plt.ylabel("Predicted solution")
# plt.show()
# 
# 
# # ## Saving and loading the trained FluidLearn model
# 
# # ### Saving the model
# 
# # In[37]:
# 
# 
# os.makedirs("./saved_models", exist_ok=True)
# 
# #Saving the model using .h5 extension
# pinn_model.save("./saved_models/trained_model_1")
# 
# 
# # In[38]:
# 
# 
# # loaded_model = keras.models.load_model("./save_models/trained_model_1",
# #                                       custom_objects={"space_dim": space_dim, 
# #                 "time_dep": time_dep, "output_dim": output_dim,
# #                  "n_hid_lay": n_hid_lay, "n_hid_nrn": n_hid_nrn,
# #                 "act_func": act_func})
# loaded_model = keras.models.load_model("./saved_models/trained_model_1",
#                                       custom_objects={"u_loss": Losses.u_loss,
#                                                      "pde_loss": Losses.pde_loss})
# 
# 
# # In[39]:
# 
# 
# new_predicted_test=loaded_model.predict(x=X_test_st)
# 
# 
# # In[40]:
# 
# 
# np.concatenate([Y_test,new_predicted_test], axis=1)
# 
# 
# # In[41]:
# 
# 
# loaded_model.fit(x=X_tr, y=[Y_tr, Y_tr], epochs=1)

