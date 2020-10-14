#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Module containing various loss functions for FluidLearn PDE solver.
"""

import numpy as np
import tensorflow as tf

__author__ = "Manu Jayadharan"
__copyright__ = "Copyright 2020, FluidLearn"
__credits__ = ["Manu Jayadharan"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Manu Jayadharan"
__email__ = "manu.jayadharan@pitt.edu"
__status__ = "Development"

def u_loss(y_true, y_pred):
    """
    Loss function to take care of boundary and initial data.
    (tensor, tensor) -> tensor
    
    arguments:
    ----------
    y_true (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde). extra 1 dim in 0+1 is to check whether the given point is 
    actually a data (bc,ic) point or a pde collocation point.
    
    y_pred (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde).
    
    output:
    -------
    tensor: of size m x o, where m and o is discussed in arguments. Essentially, loss
    is calucated only for points which have 1 on the last column of y_true. For pde
    collocation points, loss is returned as 0.
    """
    
    y_true_act = y_true[:,:-1]
    #using the last column of y_true_act to check whether the point is at the boundary
    at_boundary = tf.cast(y_true[:,-1:,],bool)
    u_sq_error = (1/2)*tf.square(y_true_act-y_pred)
    return tf.where(at_boundary, u_sq_error, 0.)

def pde_loss(y_true, y_pred):
    """
    Loss function to take care of pde constrain. 
    (tensor, tensor) -> tensor
    
    arguments:
    ----------
    y_true (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde). extra 1 dim in 0+1 is to check whether the given point is 
    actually a data (bc,ic) point or a pde collocation point.
    
    y_pred (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde).
    
    output:
    -------
    tensor: of size m x o, where m and o is discussed in arguments. Essentially, loss
    is calucated only for pde collocation points, for input data points (bc,ic etc), the loss 
    is returnd a 0.
    """
    y_true_act = y_true[:,:-1]
    #using the last column of y_true_act to check whether the point is at the boundary
    at_boundary = tf.cast(y_true[:,-1:,],bool)
    pde_sq_error = (1/2)*tf.square(y_pred)
    return tf.where(at_boundary,0.,pde_sq_error)


# In[11]:


def p_loss_mixed(y_true, y_pred):
    """
    Loss function to take care of boundary and initial data,
    when the pde is in it's mixed form.
    (tensor, tensor) -> tensor
    
    arguments:
    ----------
    y_true (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde). extra 1 dim in 0+1 is to check whether the given point is 
    actually a data (bc,ic) point or a pde collocation point.
    
    y_pred (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde).
    
    space_dim (int) : Dimension of the spatial space Omega. 
    
    output:
    -------
    tensor: of size m x o, where m and o is discussed in arguments. Essentially, loss
    is calucated only for points which have 1 on the last column of y_true. For pde
    collocation points, loss is returned as 0.
    """
    y_true_act = y_true[:,0:1]
    #using the last column of y_true_act to check whether the point is at the boundary
    at_boundary = tf.cast(y_true[:,-1:,],bool)
    p_sq_error = (1/2)*tf.square(y_true_act-y_pred)
    return tf.where(at_boundary, p_sq_error, 0.)

def p_pde_loss_mixed(y_true, y_pred):
    """
    Loss function to take care of pressure(scalar) pde constrain. 
    (tensor, tensor) -> tensor
    
    arguments:
    ----------
    y_true (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde). extra 1 dim in 0+1 is to check whether the given point is 
    actually a data (bc,ic) point or a pde collocation point.
    
    y_pred (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde).
    
    output:
    -------
    tensor: of size m x o, where m and o is discussed in arguments. Essentially, loss
    is calucated only for pde collocation points, for input data points (bc,ic etc), the loss 
    is returnd a 0.
    """
    y_true_act = y_true[:,:-1]
    #using the last column of y_true_act to check whether the point is at the boundary
    at_boundary = tf.cast(y_true[:,-1:,],bool)
    pde_sq_error = (1/2)*tf.square(y_pred)
    return tf.where(at_boundary,0.,pde_sq_error)

def u_pde_loss_mixed(y_true, y_pred):
    """
    Loss function to take care of velocity pde constrain (A vector is coming in as y_pred). 
    (tensor, tensor) -> tensor
    
    arguments:
    ----------
    y_true (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde). extra 1 dim in 0+1 is to check whether the given point is 
    actually a data (bc,ic) point or a pde collocation point.
    
    y_pred (tensor) : tensor of size m x o+1, where m is the data size and o is the 
    dimension of the output of the major neural net (same as dimension of range of 
    soluton to the pde).
    
    output:
    -------
    tensor: of size m x o, where m and o is discussed in arguments. Essentially, loss
    is calucated only for pde collocation points, for input data points (bc,ic etc), the loss 
    is returnd a 0.
    """
    y_true_act = y_true[:,:-1]
    #using the last column of y_true_act to check whether the point is at the boundary
    at_boundary = tf.cast(y_true[:,-1:,],bool)
    y_pred_tensor = tf.convert_to_tensor(y_pred)
    pde_sq_error = (1/2)*tf.square(y_pred_tensor)
    return tf.where(at_boundary,0.,tf.mean(pde_sq_error, axis=-1))


# In[ ]:




