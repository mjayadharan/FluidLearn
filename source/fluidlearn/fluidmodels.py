#!/usr/bin/env python
"""
Module containng custom Keras models and layers required for fluidlearn architecture.
"""

try:     
    import numpy
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
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

class ForwardModel(tf.keras.Model):
    """
    Model to construct FNN (Forward Neural Network) using custom Keras layers. Subclass of tf.keras.Model
    """
      
    def __init__(self, space_dim=1, time_dep=False, output_dim=1,
                 n_hid_lay=3, n_hid_nrn=20, act_func = "tanh", rhs_func = None):
        """
        space_dim (int) -> Dimension of the space Omega where the PDE is defined.
        _time_dep (bool) -> True if the problem is time dependent.
        output_dim (int) -> Dimension of the range of the solution to PDE.
        
        n_hid_layer (int) -> Number of hidden layers in the neural network.
        n_hid_nrn (int) -> Number of neurons in each hidden layer of the NN.
        
        act_func (string) -> Activation functions for each of the hidden layers. Has to
                            be one of the members of keras.activations: could be one of
                            {"tanh", "sigmoid", "elu", "relu", "exponential"}
        """
        
        super(ForwardModel, self).__init__()
        
        #Defining class atributes
        self.space_dim = space_dim
        self._time_dep = time_dep
        self.output_dim = output_dim
        self.n_hid_lay = n_hid_lay
        self.n_hid_nrn = n_hid_nrn
        
        #Block of hidden layers
        self.hidden_block = [keras.layers.Dense( self.n_hid_nrn, activation=act_func,
                                           name="dense_"+str(i+1) ) for i in range(n_hid_lay)]
        #Final output layer
        self.final_layer = keras.layers.Dense(self.output_dim,
                                         name="final_layer")        
        
        #Defining the rhs of PDE: P(u,delu) = f(x,t)
        if rhs_func != None:
            self.rhs_function = rhs_func
        else:
            self.rhs_function = lambda x: 0
        
    def findGrad(self,func,input_space):
        """
        Find gradient with respect to the domain Omega of the PDE. 
        (tensor, tensor) -> Keras.Lambda layer
        
        arguments:
        ----------
        func (tf tensor): function represented by tf tensor structure (Usually of size:
              data_size x dim_output_previous_layer). The func is usually the final output (solution u)
              coming out of a hidden layer
        
        input_space: argument with respect to which we need the partial derrivatives of func. Usually a list of 
              input arguments representing the space dimension.
        
        Output: Keras.Lambda layer. Note that output of such a lambda layer will be a list of tensors with
                each element giving partial derrivative wrt to each element in argm.
        
        See tf.Keras.Lambda and tf.gradients for more details.
        
        """
        try:
            return keras.layers.Lambda(lambda z: [tf.gradients(z[0],x_i,
                                                               unconnected_gradients='zero')
                                                  for x_i in z[1] ]) ([func, input_space])
        except Exception as e:
            raise Exception("Error occured in finding the time derrivative  lambda layer of type {} as follows: \n{}".format(type(e)),e)
          
        
    def findTimeDer(self,func,input_time):
        """
        (tensor, tensor) -> Keras.Lambda layer
        
        arguments:
        ----------
        func (tf tensor): function represented by tf tensor structure (Usually of size:
              data_size x dim_output_previous_layer). The func is usually the final output (solution u)
              coming out of a hidden layer
        
        input_time: TensorFlow tensor. This should be the element of the input list which corresponds to the time
              dimension. Used only if the problem is time_dependent.
        
        Output: Keras.Lambda layer. Note that output of such a lambda layer will be a tensor of size m x 1 
                representing the time derrivative of output func.
        
        
        See tf.Keras.Lambda and tf.gradients for more details.
        
        """
        assert (self._time_dep), "Tried taking time derrivative even though the problem is not time dependent."
        try:
            return keras.layers.Lambda(lambda z: tf.gradients(z[0],z[1],
                                                               unconnected_gradients='zero') [0]) ([func, input_time])
        except Exception as e:
            raise Exception("Error occured in find gradient lambda layer of type {} as follows: \n{} ".format(type(e)),e)
            
            
    def findLaplace(self,first_der,input_space):
        """
        (tensor, tensor) -> Keras.Lambda layer
        
        Returns lambda layer to find the laplacian of the solution to pde. 
        
        arguments:
        ----------
        first_der (tf tensor): function represented by tf tensor structure (Usually of size:
              data_size x dim_output_previous_layer). The func is 
        
        input_space: argument with respect to which we need the partial derrivatives of func. Usually a list of 
                     input arguments representing the space dimension.
        
        Output: Keras.Lambda layer. This lambda layer outputs the laplacian of solution function u.  
        
        See tf.Keras.Lambda and tf.gradients for more details.
        
        """
        try:
            # list containng diagonal entries of hessian matrix. Note that  tf.gradients 
            #returns a list of tensors and hence thats why we have  a [0] at the end of  
            #the tf.gradients fucntion as tf.gradients(func,argm) [0]
            del_sq_layer = keras.layers.Lambda( lambda z: [ tf.gradients(z[0][i], z[1][i],
                                                              unconnected_gradients='zero') [0]
                                                  for i in range(len(z[1])) ] ) ([first_der,input_space])
            return sum(del_sq_layer)
                
        except Exception as e:
            raise Exception("Error occured in find laplacian lambda layer of type {} as follows: \n{}".format(type(e)),e)
    
    #final layer representing the lhs P(x,t) of PDE P(x,t)=0
    def findPdeLayer(self, laplacian, input_arg, time_der=0):
        """
        (tensor, tensor, tensor) -> Keras.Lambda layer
        
        Returns lambda layer to find the actual pde P(u,delu,x,t) such that P(u,delu,x,t)=0. 
        
        arguments:
        ----------
        laplacian (tf tensor): laplacian with respect to space dim .
        
        input_arg: list of inputs corresponding to both space and time dimension. Last elemetn of 
                   the list corresponds to the temporal dimension.
        
        Output: Keras.Lambda layer. This lambda layer outputs the PDE P(u,delu, x,t).  
        
        See tf.Keras.Lambda and tf.gradients for more details.
        
        """
        try:
#             return keras.layers.Lambda(lambda z: z[0] - z[1] - tf.sin(z[2][0]+z[2][1]) - 
#                                        2*z[2][2]*tf.sin(z[2][0]+z[2][1])) ([time_der, laplacian, input_arg])
            return keras.layers.Lambda(lambda z: z[0] - z[1] - self.rhs_function(input_arg)) ([time_der, laplacian, input_arg])
        except Exception as e:
            raise Exception("Error occured in finding pde  lambda layer of type {} as follows: \n{}".format(type(e)),e)
    
    def get_config(self):
        #getting basic config using the parent model class
        base_config = super().get_config()
        return {**base_config, "space_dim": self.space_dim, 
                "_time_dep": self._time_dep, "output_dim": self.output_dim,
                 "n_hid_lay": self.n_hid_lay, "n_hid_nrn": self.n_hid_nrn,
                "act_func": self.act_func }
    
    def from_config(self, config, custom_objects):
        super().from_config(config)
    
    def call(self, inputs, training=False):
        """
        Call function which wll be used while training, prediciton and evaluation of the ForwardModel. 
        
        arguments:
        ----------
        inputs (list of tensors) -> last element of the list corresponds to temporal diimension if 
                                    self._time_dep = True. If possible, always feed the data from the 
                                    data processing method in flowDataProcess module.
        training (bool) -> True if calling the function for training. False for prediction and evaluation. 
                           Value of triainng will be automatically taken care of by Keras. 
        
        Note that inputs should always be given as a list with the last element of the list representing the 
        dimension corresponding to time.
        """
        if self._time_dep:
            try:
                assert(len(inputs) > 1)
                input_space = inputs[:-1]
                input_time = inputs[-1]
            except Exception as e:
                raise Exception("Error occured while separating spacial and temporal data from inputs,                make sure that spacio-temporal data is being used to for training and                 x=[space_dim1,..,space_dimn,time_dim]. More details on error below:\n", type(e), e)
        else:
            input_space = inputs
        
        #concatening all the input data (space and time dimensions) making it 
        #read to be passed to the hidden layers
        hidden_output = keras.layers.concatenate(inputs) 
        
        #hidden layers
        for layer_id in range(self.n_hid_lay):
            hidden_output = self.hidden_block[layer_id] (hidden_output)

        
        #output layer, this is typically the solution function
        output_layer = self.final_layer(hidden_output)
        
        if training:
            #pde specific layers
            grad_layer = self.findGrad(output_layer, input_space)
            laplace_layer = self.findLaplace(grad_layer, input_space)
            if self._time_dep: 
                time_der_layer = self.findTimeDer(output_layer, input_time)
            else:
                time_der_layer=0
            pde_layer = self.findPdeLayer(laplace_layer, inputs, time_der_layer)

            return output_layer, pde_layer
        
        elif not training: #only outputting the function value if not tranining.
                return output_layer


        

