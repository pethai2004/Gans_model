import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils import DefaultConfig

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
    # x0 = tf.cast(x0, tf.float32)
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
    # x0 = tf.cast(x0, tf.float32)
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
    # x0 = tf.cast(x0, tf.float32)
    return x0

def get_noise_out(x0, mean=0., stddev=1.):
    shape_x = x0.shape
    with tf.name_scope("add_noise"):
        x0 = keras.layers.GaussianNoise(stddev)(x0)
    # x0 = tf.cast(x0, dtype=tf.float32)
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
    # x0 = tf.cast(x0, tf.float32)
    
    return x0

class BaseGenerator:

    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=200, seed=5005, strategy_scope=None, name="BaseGenerator"):
        self.img_size = img_size
        self.targ_img_size = targ_img_size
        self.z_dim = z_dim
        self.seed = seed
        self.strategy_scope = strategy_scope
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        self.initialized = False
        self.inputer = None
        self.outputer = None
        self.model = None
        self.id = None
        self.name = name
        self.extendable = True
        self.configbuild = {}

    def initialize_base(self):
        self.initialized = True
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._initialize_base()
        else:
            self._initialize_base()

    def extent_model(self):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self.extend_model()
        else:
            self.extend_model()

    def _extend_model(self):
        raise NotImplementedError

    def _initialize_base(self):
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def forward_mapping(self, inputs):
        '''forward mapping layer only'''
        raise NotImplementedError
    
    def get_mapping_weights():
        raise NotImplementedError
    
    def forward_model(self, inputs, training=True):
        inputs = self.model(inputs, training=training)
        return tf.cast(inputs, dtype=tf.float32)
    
    def add_noise(self, inputs):
        return inputs
    
    def set_latent(self, trainable=False):
        '''Freezing trainable weights for latent mapping'''
        raise NotImplementedError

class BaseDiscriminator:

    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=128, seed=1010, strategy_scope=None, name="DISC01"):
        self.img_size = img_size
        self.targ_img_size = targ_img_size
        self.z_dim = z_dim
        self.seed = seed
        self.strategy_scope = strategy_scope
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        self.initialized = False
        self.inputer = None
        self.outputer = None
        self.model = None
        self.id = None
        self.name = name
        self.configbuild = {}

    def initialize_base(self):
        self.initialized = True
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._initialize_base()
        else:
            self._initialize_base()

class Generator_v0(BaseGenerator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=128, seed=1010, strategy_scope=None, name="GEN01",
                up_type="deconv", out_units=256, dense_units=256):
        '''
        Input : 
            image_size : (tuple) target image size at starting training (before first extent model)
            targ_img_size : (tuple) target image size at the end of training
        '''
        super(Generator_v0, self).__init__(img_size, targ_img_size, z_dim, seed, strategy_scope, name)
        assert up_type in ["deconv", "upsample"], "up_type must be one of deconv or upsample"
        assert img_size[0] == img_size[1], "img_size must be square"
        assert targ_img_size[0] == targ_img_size[1], "targ_img_size must be square"
        self.up_type = up_type
        self.dense_units = dense_units 
        self.num_layers = 6
        self.out_units = out_units
        self.blocks = []
        self.cur_img_size = img_size
    
    def _initialize_base(self):
        '''Initialize model for based layer map and conv'''
        self.inputer = keras.layers.Input(shape=(self.z_dim))
        x1 = latent_mapping(self.inputer, num_layers=self.configbuild["num_layers"], units=self.configbuild["dense_units"],
                out_units=self.configbuild["out_units"], activation=self.configbuild["dense_act"], initializer=self.initializer, max_norm=None, batch_norm=True)
        
        x1 = keras.layers.Reshape(target_shape=(1, 1, self.configbuild["out_units"]), name='reshape_mapper')(x1)

        if self.up_type == "deconv":
            for i_layer in self.configbuild["BaseFilters"]:
                x1 = get_deconv(x1, i_layer, self.configbuild["kernel_size"], strides=(2, 2), padding="same", activation=self.configbuild["conv_act"], initializer=self.initializer, 
                        add_noise=None, max_norm=None, batch_norm=True)
        
        elif self.up_type == "upsample":
            for i_layer in self.configbuild["BaseFilters"]:
                x1 = get_upsampling(x1, i_layer, self.configbuild["kernel_size"], strides=(1, 1), padding="same", up_size=(2, 2), activation=self.configbuild["conv_act"], 
                    interpolation="nearest", initializer=self.initializer, add_noise=None, batch_norm=True)
                
        self.outputer = keras.layers.Conv2D(3, self.configbuild["kernel_size"], strides=(1, 1), padding="same", activation=self.self.configbuild["out_act"], use_bias=True)(x1)
        assert tf.reduce_prod(x1.shape[1:-1]) == tf.reduce_prod(self.img_size), "mapping blocks do not correctly output shape, "\
        "must provide filters with length of {}".format(int(np.log2(self.img_size[0])))
        self.model = keras.Model(self.inputer, self.outputer)
        
    def _extend_model(self, filters=200, up=(1, 1), noise=get_noise_out):
        '''extent conv block'''
        if not self.initialized:
            raise ValueError("Generator model not initialized")
        self.cur_img_size = (self.cur_img_size[0] * up[0], self.cur_img_size[0] * up[1])
        last_shape = self.outputer.shape
        
        if self.up_type == "deconv":
            self.outputer = get_deconv(self.outputer, filters, self.configbuild["kernel_size"], strides=up, padding="same", activation='selu', initializer=self.initializer, 
                add_noise=noise, max_norm=None, batch_norm=True)
        
        elif self.up_type =="upsample":
            self.outputer = get_upsampling(self.outputer, filters, self.configbuild["kernel_size"], strides=(1, 1), padding="same", up_size=up, activation='selu', 
                interpolation="nearest", initializer=self.initializer, add_noise=noise, batch_norm=True)
        
        if up[0] > 1:
            self.outputer = keras.layers.Conv2D(3, (5, 5), strides=(1, 1), padding="same", activation=None, use_bias=True)(self.outputer)
            assert tf.reduce_prod(self.outputer.shape[1:-1]) == tf.reduce_prod(self.cur_img_size), "extended model do not correctly output shape"\
            "must provide filters with length of {}".format(int(np.log2(self.cur_img_size[0])))
        self.model = keras.Model(self.inputer, self.outputer)
        print("extended model from size of", last_shape, "to", self.outputer.shape)
    
    def auto_extend(self, filters=[300, 300, 300], noise=get_noise_out):
        for ifilt in filters[:-1]:
            self._extend_model(ifilt, up=(1, 1))
        self._extend_model(filters[-1], up=(2, 2), noise=noise)

class Discriminator_v0(BaseDiscriminator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=200, seed=5005, strategy_scope=None, 
            name="BaseDiscriminator", extendable=False):
        '''
        Input : 
            img_size : (tuple) target image size at starting training (before first extent model)
            targ_img_size : (tuple) target image size at the end of training
            non_extendable : (bool) specify whether this discriminator is fixed input or applicable to input pixel
        '''
        super(Discriminator_v0, self).__init__(img_size, targ_img_size, z_dim, seed, strategy_scope, name)
        assert img_size[0] == img_size[1], "img_size must be square"
        assert targ_img_size[0] == targ_img_size[1], "targ_img_size must be square"
        self.out_units = 200
        self.cur_img_size = self.img_size
        self.extendable = extendable
        self.blocks = []

    def extend_size(self):
        self.cur_img_size = (self.cur_img_size[0] * 2, self.cur_img_size[1] * 2)

    def _initialize_base(self, filters, units_dense, act_out=None):
        '''Initialize model for based layer map and conv'''
        self.initialized = True
        if not self.extendable:
            cur_in = keras.layers.Input(shape=(*self.targ_img_size, 3))
        else :
            cur_in = keras.layers.Input(shape=(*self.cur_img_size, 3))
        x1 = get_downsampling_conv(cur_in, filters[0], kernel_size=(5, 5), strides=(1, 1), padding="same", 
                activation='relu', pooling=(2, 2), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
        
        for filts in filters[1:]:
            x1 = get_downsampling_conv(x1, filts, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                activation='relu', pooling=(2, 2), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
        x1 = keras.layers.Flatten(name="flatten_d_conv")(x1)
        for ui in units_dense:
            x1 = keras.layers.Dense(ui, activation='selu', use_bias=True, kernel_initializer=self.initializer)(x1)
        cur_out = keras.layers.Dense(1, activation=act_out)(x1)
        if not self.extendable:
            m = keras.Model(cur_in, cur_out)
            self.blocks.append(m)
        else: self.model = keras.Model(cur_in, cur_out)
            
    def forward_model(self, inputs, training=True):
        
        if not self.extendable: # fixed model, upsample first
            shapex = inputs.shape[1:-1]
            multiplier = int(self.targ_img_size[0] / shapex[0])
            if multiplier > 1:
                inputs = keras.layers.UpSampling2D((multiplier, multiplier), interpolation="nearest")(inputs)     
            return self.model(inputs, training=training)
        else:
            for ly in reversed(self.blocks):
                inputs = ly(inputs, training=training)
            return inputs

    def auto_extend(self, filters):
        assert self.cur_img_size != self.targ_img_size, "cannot extend model, current input shape exceed targ_img_size"
        if self.extendable:
            self.extend_size()
            cur_in = keras.layers.Input(shape=(*self.cur_img_size, 3))
            x1 = get_downsampling_conv(cur_in, filters[0], kernel_size=(5, 5), strides=(1, 1), padding="same", 
                        activation='relu', pooling=(1, 1), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
            for ifilt in filters[1:]:
                x1 = get_downsampling_conv(x1, ifilt, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                    activation='relu', pooling=(1, 1), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
                
            cur_out = get_downsampling_conv(x1, 3, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                        activation='relu', pooling=(2, 2), initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True)
            m = keras.Model(cur_in, cur_out)
            self.blocks.append(m)
        else:
            raise ValueError("This Discriminator is not extendable")
    
    def get_model(self, get_with_functional=True):
        assert self.extendable, "This Discriminator is not extendable"
        assert self.cur_img_size == self.targ_img_size, 'get_model cannot be called if model still not extend to targ_img_size'
        if get_with_functional: U_layers = list(reversed(self.blocks))
        else:
            U_layers = []
            for msl in reversed(self.blocks):
                for ms in msl.layers: U_layers.append(ms)
        self.inputer = keras.layers.Input(shape=(*self.targ_img_size, 3))
        xi = U_layers[0](self.inputer)
        for J_layers in U_layers[1:]:
            xi = J_layers(xi)
            
        return keras.Model(self.inputer, xi)

class Discriminator_v1(BaseDiscriminator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=None, z_dim=128, seed=1010, strategy_scope=None, name="DISC02"):
        assert targ_img_size is None, "targ_img_size must not be specified"
        super(Discriminator_v1, self).__init__(img_size, targ_img_size, z_dim, seed, strategy_scope, name)
        self.extendable = False

    def _initialize_base(self):
        input_dims = keras.layers.Input(shape=(*self.img_size, 3))
        
        xk = keras.layers.Conv2D(200, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                     kernel_initializer=self.initializer, activation='relu')(input_dims)
        xk = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Conv2D(300, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
                                      kernel_initializer=self.initializer, activation='relu')(xk)
        xk = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding="same")(xk)
        xk = keras.layers.BatchNormalization()(xk)
        
        xk = keras.layers.Conv2D(300, (5, 5), strides=(1, 1), padding='same', use_bias=True, 
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
        
    def forward_model(self, input_tensor, training=True):
        return self.model(input_tensor, training=training)
    
class Generator_v1(BaseGenerator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=None, z_dim=200, seed=5005, strategy_scope=None, name="BaseGenerator"):
        assert targ_img_size is None, "targ_img_size must not be specified"
        super(Generator_v1, self).__init__(img_size, targ_img_size, z_dim, seed, strategy_scope, name)
        self.extendable = False
        self.out_latent = 300

    def _initialize_base(self):
        input_dims = keras.layers.Input(shape=(self.z_dim, ))
        
        xk = keras.layers.Dense(256, activation='relu')(input_dims)
        xk = keras.layers.Dense(self.out_latent)(xk)
        xk = keras.layers.Reshape((1, 1, self.out_latent))(xk)
        xk = keras.layers.Conv2DTranspose(300, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(400, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(400, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Conv2DTranspose(500, (5, 5), strides=(1, 1), padding='same', activation='relu', 
                                              use_bias=True, kernel_initializer=self.initializer)(xk)
        xk = keras.layers.Flatten()(xk)
        xk = keras.layers.Dense(200)(xk)
        xk = keras.layers.Dense(tf.reduce_prod(self.img_size) * 3)(xk)
        xk = keras.layers.Reshape(self.img_size)(xk)
        
        self.model = keras.Model(input_dims, xk)
        
    def forward_model(self, input_tensor, training=True):
        return self.model(input_tensor, training=training)

