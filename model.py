import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_deconv(x0, filters, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', initializer=None, 
                   add_noise=None, max_norm=None, batch_norm=True):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("deconv_up"):
        x0 = keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding=padding, activation=activation, 
                                                  use_bias=True, kernel_initializer=initializer)(x0)
    if add_noise:
        x0 = add_noise(x0)
    if max_norm is not None:
        x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0)(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization()(x0)
    x0 = tf.cast(x0, tf.float32)
    return x0

def get_upsampling(x0, filters, kernel_size, strides, padding="same", up_size=(2, 2), 
                       activation='relu', interpolation="nearest", initializer=None, add_noise=None, batch_norm=True):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("upsampling"):
        x0 = keras.layers.UpSampling2D(up_size, interpolation=interpolation)(x0)
        x0 = keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=True, 
                                         kernel_initializer=initializer, activation=activation)(x0)
    if add_noise:
        x0 = add_noise(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization()(x0)
    x0 = tf.cast(x0, tf.float32)
    return x0

def latent_mapping(x0, num_layers=8, units=128, out_units=1024, activation='selu', initializer=None, max_norm=None, batch_norm=True):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("map_latent_z"):
        for n in range(num_layers - 1):
            x0 = keras.layers.Dense(units, activation=activation, kernel_initializer=initializer, name='map_z' + str(n))(x0)
            if max_norm is not None:
                x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0)(x0)
        x0 = keras.layers.Dense(out_units, activation=activation, kernel_initializer=initializer, name='map_z'+str(num_layers-1))(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization()(x0)
    x0 = tf.cast(x0, tf.float32)
    return x0

def get_noise_out(x0, mean=0., stddev=1.):
    shape_x = x0.shape
    with tf.name_scope("add_noise"):
        x0 = keras.layers.GaussianNoise(stddev)(x0)
    x0 = tf.cast(x0, dtype=tf.float32)
    return x0

def get_downsampling_conv(x0, filters, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', pooling=(2, 2), initializer=None, 
                   add_noise=None, max_norm=None, batch_norm=True):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("down_samping"):
        x0 = keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, activation=activation, 
                                                  use_bias=True, kernel_initializer=initializer)(x0)
    if add_noise:
        x0 = add_noise(x0)
    if max_norm is not None:
        x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0)(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization()(x0)
    if pooling is not None:
        x0 = keras.layers.MaxPooling2D(pool_size=pooling, strides=None, padding="same")(x0)
    x0 = tf.cast(x0, tf.float32)
    
    return x0

def double_block(x0, filters=300):
    x0 = get_upsampling(x0, filters=filters, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', initializer=None, 
                   add_noise=None, max_norm=None, batch_norm=True)
    x1 = keras.layers.Flatten()(x0)

    x1 = keras.layers.Dense(128, activation=None)(x1)
    @tf.function
    def bl_g(x0):
        gk_Ks = tf.gradient(x1, x)

class Generator_v1:
    
    def __init__(self, image_size=(32, 32), targ_img_size=(128, 128), up_type="deconv", latent_space=128, seed=1010, strategy_scope=None):
        '''
        Input : 
            image_size : (tuple) target image size at starting training (before first extent model)
            targ_img_size : (tuple) target image size at the end of training
        '''
        super().__init__()
        self.image_size = image_size 
        self.targ_img_size = targ_img_size
        self.blocks = []
        self.latent_space = latent_space
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        assert up_type in ["deconv", "upsample"], "up_type must be one of deconv or upsample"
        self.up_type = up_type
        self.start = False
        self.out_units = 200
        self.model = None
        self.inputer = None
        self.outputer = None
        self.cur_img_size = self.image_size
        self.strategy_scope = strategy_scope    
        self.num_layers = 4
        self.dense_units = 256
        self.name = "Gen001"

    def initialize_base(self, filters):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._initialize_base(filters)
        else:
            self._initialize_base(filters)

    def extent_model(self, filters, up=(1, 1), noise=get_noise_out):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self.extend_model(filters, up, noise)
        else:
            self.extend_model(filters, up, noise)

    def _initialize_base(self, filters):
        '''Initialize model for based layer map and conv'''
        self.start = True
        self.inputer = keras.layers.Input(shape=(self.latent_space))
        x1 = latent_mapping(self.inputer, num_layers=self.num_layers, units=self.dense_units, out_units=self.out_units, activation='selu', 
                                initializer=self.initializer, max_norm=None, batch_norm=True)
        
        x1 = keras.layers.Reshape(target_shape=(1, 1, self.out_units), name='reshape_mapper')(x1)

        if self.up_type == "deconv":
            for i_layer in filters:
                x1 = get_deconv(x1, i_layer, (5, 5), strides=(2, 2), padding="same", activation='selu', initializer=self.initializer, 
                        add_noise=None, max_norm=None, batch_norm=True)
        
        elif self.up_type =="upsample":
            for i_layer in filters:
                x1 = get_upsampling(x1, i_layer, (5, 5), strides=(1, 1), padding="same", up_size=(2, 2), activation='selu', 
                    interpolation="nearest", initializer=self.initializer, add_noise=None, batch_norm=True)
                
        self.outputer = keras.layers.Conv2D(3, (5, 5), strides=(1, 1), padding="same", activation=None, use_bias=True)(x1)
        assert tf.reduce_prod(x1.shape[1:-1]) == tf.reduce_prod(self.image_size), "mapping blocks do not correctly output shape"
        self.model = keras.Model(self.inputer, self.outputer)
        
    def _extend_model(self, filters=200, up=(1, 1), noise=get_noise_out):
        '''extent conv block'''
        if not self.start:
            self.initialize_base()
        self.cur_img_size = (self.cur_img_size[0] * up[0], self.cur_img_size[0] * up[1])
        last_shape = self.outputer.shape
        
        if self.up_type == "deconv":
            self.outputer = get_deconv(self.outputer, filters, (5, 5), strides=up, padding="same", activation='selu', initializer=self.initializer, 
                add_noise=noise, max_norm=None, batch_norm=True)
        
        elif self.up_type =="upsample":
            self.outputer = get_upsampling(self.outputer, filters, (5, 5), strides=(1, 1), padding="same", up_size=up, activation='selu', 
                interpolation="nearest", initializer=self.initializer, add_noise=noise, batch_norm=True)
        
        if up[0] > 1:
            self.outputer = keras.layers.Conv2D(3, (5, 5), strides=(1, 1), padding="same", activation=None, use_bias=True)(self.outputer)
            assert tf.reduce_prod(self.outputer.shape[1:-1]) == tf.reduce_prod(self.cur_img_size), "extended model do not correctly output shape"
        self.model = keras.Model(self.inputer, self.outputer)
        print("extended model from size of", last_shape, "to", self.outputer.shape)
    
    def auto_extend(self, filters=200, num_extend=3, noise=get_noise_out):
        for i in range(num_extend - 1):
            self.extend_model(filters, up=(1, 1))
        self.extend_model(filters, up=(2, 2), noise=noise)
        
    def forward_mapping(self, inputs):
        '''forward mapping layer only'''
        raise NotImplementedError
    
    def get_mapping_weights():
        raise NotImplementedError
    
    def forward_model(self, inputs, training=True):
        self.model(inputs, training=training)
        return inputs
    
    def add_noise(self, inputs):
        
        return inputs
    
    def set_latent(self, trainable=False):
        '''Freezing trainable weights for latent mapping'''
        raise NotImplementedError

class Discriminator_v1:
    
    def __init__(self, image_size=(32, 32), targ_img_size=(128, 128), non_extendable=True, seed=1010, strategy_scope=None):
        '''
        Input : 
            image_size : (tuple) target image size at starting training (before first extent model)
            targ_img_size : (tuple) target image size at the end of training
            non_extendable : (bool) specify whether this discriminator is fixed input or applicable to input pixel
        '''
        super().__init__()
        self.image_size = image_size 
        self.targ_img_size = targ_img_size
        self.blocks = []
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        self.start = False
        self.out_units = 200
        self.model = None
        self.inputer = None
        self.cur_img_size = self.image_size
        self.non_extendable = non_extendable
        self.outputer = None
        self.strategy_scope = strategy_scope
        self.name = "disc001"

    def initialize_base(self, filters, units_dense, act_out=None):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._initialize_base(filters, units_dense, act_out)
        else:
            self._initialize_base(filters, units_dense, act_out)

    def extent_model(self, filters, up, noise=get_noise_out):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self.extend_model(filters, up, noise)
        else:
            self.extend_model(filters, up, noise)

    def _initialize_base(self, filters, units_dense, act_out=None):
        '''Initialize model for based layer map and conv'''
        self.start = True
        
        if self.non_extendable:
            self.inputer = keras.layers.Input(shape=(*self.targ_img_size, 3))
            x1 = get_downsampling_conv(self.inputer, filters[0], kernel_size=(5, 5), strides=(1, 1), padding="same", 
                    activation='relu', pooling=(2, 2), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
            
            for filts in filters[1:]:
                x1 = get_downsampling_conv(x1, filts, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                    activation='relu', pooling=(2, 2), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
            x1 = keras.layers.Flatten(name="flatten_d_conv")(x1)
            for ui in units_dense:
                x1 = keras.layers.Dense(ui, activation='selu', use_bias=True, kernel_initializer=self.initializer)(x1)
            self.outputer = keras.layers.Dense(1, activation=act_out)(x1)
            self.model = keras.Model(self.inputer, self.outputer)
            
        else:
            raise NotImplementedError
            
    def _extend_model(self, filters=200, up=(1, 1), noise=get_noise_out):
        '''extent conv block'''
        assert not self.non_extendable, "non_extendable Discriminator cannot be called by extend_model"
        raise NotImplementedError
        
    def auto_extend(self, filters=200, num_extend=3, noise=get_noise_out):
        
        raise NotImplementedError
    
    def forward_model(self, inputs, training=True):
        
        if self.non_extendable: # fixed model, upsample first
            shapex = inputs.shape[1:-1]
            multiplier = int(self.targ_img_size[0] / shapex[0])
            if multiplier > 1:
                inputs = keras.layers.UpSampling2D((multiplier, multiplier), interpolation="nearest")(inputs)
                
        return self.model(inputs, training=training)

class Discriminator_v0:
    
    def __init__(self, image_shape=(28, 28, 1)):
        self.image_shape = image_shape
        self.model_layer = []
        self.model = None
        self.initializer = keras.initializers.RandomNormal(stddev=0.01)
        self.name = "disc002"

    def initialize_base(self):
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
        self.name = "gen002"

    def initialize_base(self):
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

defualt_set_G = None
defualt_set_D = None

