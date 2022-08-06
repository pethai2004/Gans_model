import tensorflow as tf
from tensorflow import keras
import numpy as np

class Discriminator_v0:
    
    def __init__(self, image_shape=(28, 28, 1)):
        self.image_shape = image_shape
        self.model_layer = []
        self.model = None
        self.initializer = keras.initializers.RandomNormal(stddev=0.01)
        
    def initialize_model(self):
        input_dims = keras.layers.Input(shape=self.image_shape)
        
        xk = keras.layers.Conv2D(100, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                     kernel_initializer=self.initializer, activation='relu')(input_dims)
        xk = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Conv2D(250, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                      kernel_initializer=self.initializer, activation='relu')(xk)
        xk = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Conv2D(250, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                     kernel_initializer=self.initializer, activation='relu')(xk)
        xk = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Conv2D(400, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                     kernel_initializer=self.initializer, activation='relu')(xk)
        xk = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Flatten()(xk)
        xk = keras.layers.Dense(200)(xk)
        xk = keras.layers.Dense(1, activation=None)(xk)
        
        self.model = keras.Model(input_dims, xk)
        
    def forward_model(self, input_tensor):
        return self.model(input_tensor)
    
    def extent_model(self):
        raise NotImplementedError
    
class Generator_v0:
    
    def __init__(self, latent_z=128, out_latent=128, image_shape=(28, 28, 1)):
        self.out_latent = out_latent
        self.latent_z = latent_z
        self.image_shape = image_shape
        self.model_layer = []
        self.model = None 
        self.initializer = keras.initializers.RandomNormal(stddev=0.01)
        
    def initialize_model(self):
        input_dims = keras.layers.Input(shape=self.latent_z)
        
        xk = keras.layers.Dense(128, activation='relu')(input_dims)
        xk = keras.layers.Dense(self.out_latent)(xk)
        xk = keras.layers.Reshape((1, 1, self.out_latent))(xk)
        xk = keras.layers.Conv2DTranspose(100, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(200, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(300, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(500, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Flatten()(xk)
        xk = keras.layers.Dense(200)(xk)
        xk = keras.layers.Dense(tf.reduce_prod(self.image_shape))(xk)
        xk = keras.layers.Reshape(self.image_shape)(xk)
        
        self.model = keras.Model(input_dims, xk)
        
    def forward_model(self, input_tensor):
        return self.model(input_tensor)
    
    def extent_model(self):
        raise NotImplementedError 

def cov2d_block(x, filters=128, act='relu', kern=(5, 5), strd=(1, 1), pad='same', BN=False, DO=0.):
    initializer = keras.initializers.RandomNormal(stddev=0.01)
    x = keras.layers.Conv2D(filters, kern, strd, pad, activation=act, use_bias=True, kernel_initializer=initializer)(x)
    if BN:
        x = keras.layers.BatchNormalization()(x)
    if DO > 0.:
        x = keras.layers.Dropout(D)(x)
    return x

def mp_conv2d_block(x, filters=128, act='relu', kern=(5, 5), strd=(1, 1), pad='same', DO=1.):
    
    initializer = keras.initializers.RandomNormal(stddev=0.01)
    x = keras.layers.Conv2D(filters, kern, strd, pad, activation=act, use_bias=True, kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(DO)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(x)
    
    return x

def deconv2d_block(x, filters=64, act='relu', kern=(5, 5), strd=(1, 1), pad='same', BN=False, DO=0.):
    initializer = RandomNormal(stddev=0.01)
    
    x = keras.layers.Conv2DTranspose(filters, kern, strd, pad, activation=act, use_bias=True, 
                                         kernel_initializer=initializer)(x)
    if BN:
        x = keras.layers.BatchNormalization()(x)
    if DO > 0.:
        x = keras.layers.Dropout(DO)(x)
    return x

def dense_block(x, input_dim=256, return_shape=256):
    initializer = RandomNormal(stddev=0.01)
    
    x = keras.layers.Dense(return_shape, activation='relu',kernel_initializer=initializer, use_bias=True)(x)
    return x