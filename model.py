import tensorflow as tf
from tensorflow import keras
import numpy as np
#TODO : change processing in forward_model call in generator to be preprocessed in data_handler instead 

def get_deconv(x0, filters, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', initializer=None, 
                   add_noise=None, max_norm=None, batch_norm=True, name_suffix=""):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("deconv_up"):
        x0 = keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding=padding, activation=activation, 
                                                  use_bias=True, kernel_initializer=initializer, name="ConvTrans"+name_suffix)(x0)
    if add_noise:
        x0 = add_noise(x0, name_suffix=name_suffix)
    if max_norm is not None:
        x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0, name="MinMax"+name_suffix)(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization(name="BatchNorm"+name_suffix)(x0)
    # x0 = tf.cast(x0, tf.float32)
    return x0

def get_upsampling(x0, filters, kernel_size, strides, padding="same", up_size=(2, 2), 
                       activation='relu', interpolation="nearest", initializer=None, add_noise=None, batch_norm=True, name_suffix=""):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("upsampling"):
        x0 = keras.layers.UpSampling2D(up_size, interpolation=interpolation, name="Upsampling"+name_suffix)(x0)
        x0 = keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=True, 
                                         kernel_initializer=initializer, activation=activation, name="Conv2D"+name_suffix)(x0)
    if add_noise:
        x0 = add_noise(x0, name_suffix=name_suffix)
    if batch_norm:
        x0 = keras.layers.BatchNormalization(name="BatchNorm"+name_suffix)(x0)
    # x0 = tf.cast(x0, tf.float32)
    return x0

def latent_mapping(x0, num_layers=8, units=128, out_units=1024, activation='selu', initializer=None, max_norm=None, batch_norm=True, namemap="map_z", name_suffix=""):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("map_latent_z"):
        for n in range(num_layers - 1):
            x0 = keras.layers.Dense(units, activation=activation, kernel_initializer=initializer, name=namemap+str(n))(x0)
            if max_norm is not None:
                x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0, name="MinMax"+name_suffix)(x0)
        x0 = keras.layers.Dense(out_units, activation=activation, kernel_initializer=initializer, name=namemap+str(num_layers-1))(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization(name="BatchNorm"+name_suffix)(x0)
    # x0 = tf.cast(x0, tf.float32)
    return x0

def get_noise_out(x0, stddev=0.1, name_suffix=""):
    with tf.name_scope("add_noise"):
        x0 = keras.layers.GaussianNoise(stddev, name="NoiseOut"+name_suffix)(x0)
    # x0 = tf.cast(x0, dtype=tf.float32)
    return x0

def get_downsampling_conv(x0, filters, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', pooling=(2, 2), initializer=None, 
                   add_noise=None, max_norm=None, batch_norm=True, name_suffix=""):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(stddev=0.01)
    with tf.name_scope("down_samping"):
        x0 = keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, activation=activation, 
                                                  use_bias=True, kernel_initializer=initializer, name='Conv2D'+name_suffix)(x0)
    if add_noise:
        x0 = add_noise(x0, name_suffix=name_suffix)
    if max_norm is not None:
        x0 = keras.constraints.MinMaxNorm(min_value=-max_norm, max_value=max_norm, rate=1.0, axis=0, name="MinMax"+name_suffix)(x0)
    if batch_norm:
        x0 = keras.layers.BatchNormalization(name="BatchNorm"+name_suffix)(x0)
    if pooling is not None:
        x0 = keras.layers.MaxPooling2D(pool_size=pooling, strides=None, padding="same", name="MaxPool"+name_suffix)(x0)
    # x0 = tf.cast(x0, tf.float32)
    return x0

class NameCaller(object):
    def __init__(self):
        self.num = 0
    @property
    def n(self):
        self.num += 1
        return str(self.num).zfill(3)
        
class BaseGenerator:

    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=200, seed=5005, strategy_scope=None, name="BaseGenerator", configbuild={}):
        self.img_size = img_size
        self.targ_img_size = targ_img_size
        self.z_dim = z_dim
        self.seed = seed
        self.strategy_scope = strategy_scope
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        self.initialized = False
        self.model = None
        self.name = name
        self.extendable = True
        self.configbuild = configbuild

    def initialize_base(self):
        assert self.initialized == False, "Generator already initialized"
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

    def auto_extend(self):
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._auto_extend()
        else:
            self._auto_extend()

    def _extend_model(self): raise NotImplementedError

    def _initialize_base(self): raise NotImplementedError

    def _auto_extend(self): raise NotImplementedError
    
    def forward_mapping(self, inputs):
        '''forward mapping layer only'''
        raise NotImplementedError
    
    def get_mapping_weights(self): raise NotImplementedError
    
    def forward_model(self, inputs, training=True):
        inputs = self.model(inputs, training=training)
        return tf.cast(inputs, dtype=tf.float32)
    
    def add_noise(self, inputs):
        return inputs
    
    def set_latent(self, trainable=False):
        '''Freezing trainable weights for latent mapping'''
        raise NotImplementedError

    def __repr__(self):
        return self.name

class BaseDiscriminator:

    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), seed=1010, strategy_scope=None, name="DISC01", configbuild={}):
        self.img_size = img_size
        self.targ_img_size = targ_img_size
        self.seed = seed
        self.strategy_scope = strategy_scope
        self.initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
        self.initialized = False
        self.model = None
        self.id = None
        self.name = name
        self.configbuild = configbuild
        self.model = None

    def initialize_base(self):
        assert self.initialized == False, 'BaseDiscriminator already initialized'
        self.initialized = True
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._initialize_base()
        else:
            self._initialize_base()

    def extent_model(self):
        assert self.img_size != self.targ_img_size, "provided img_size is equal to targ_img_size, so it cannot be extended"
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self.extend_model()
        else:
            self.extend_model()

    def auto_extend(self):
        assert self.img_size != self.targ_img_size, "provided img_size is equal to targ_img_size, so it cannot be extended"
        if self.strategy_scope is not None:
            with self.strategy_scope.scope():
                self._auto_extend()
        else:
            self._auto_extend()
    
    def _extend_model(self): raise NotImplementedError

    def _initialize_base(self): raise NotImplementedError

    def _auto_extend(self): raise NotImplementedError

class Generator_v0(BaseGenerator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), z_dim=128, seed=1010, strategy_scope=None, name="GEN01",
                up_type="deconv", apply_resize=False, configbuild={}):
        '''
        Input : 
            image_size : (tuple) target image size at starting training (before first extent model).
            targ_img_size : (tuple) target image size at the end of training.
            apply_resize : (bool) whether to resize the img_size to the targ_img_size if is not equal in the forward call.
        '''

        super(Generator_v0, self).__init__(img_size, targ_img_size, z_dim, seed, strategy_scope, name, configbuild)
        assert up_type in ["deconv", "upsample"], "up_type must be one of deconv or upsample"
        assert img_size[0] == img_size[1], "img_size must be square"
        assert targ_img_size[0] == targ_img_size[1], "targ_img_size must be square"
        self.up_type = up_type
        self.blocks = []
        self.cur_img_size = img_size
        self.apply_resize = apply_resize
        self.blocks = []
        self.suff = NameCaller()
        self.optimizer = None
        self.cur_trainable = None # use just for creation of Variable that hold outside of tf.function
        self._num_call_forw = 0
        self._n_call = [len(k_) for k_ in self.configbuild["filters"]]
        self._n_call.insert(0, 1)

    def _initialize_base(self):
        '''Initialize model for based layer map and conv'''
        inputer = keras.layers.Input(shape=(self.z_dim, ), name="BaseInput"+self.suff.n)
        x1 = latent_mapping(inputer, num_layers=self.configbuild["num_layers"], units=self.configbuild["dense_units"],
                out_units=self.configbuild["out_units"], activation=self.configbuild["dense_act"], initializer=self.initializer, max_norm=None, 
                batch_norm=True, namemap="map_z", name_suffix="")
        
        x1 = keras.layers.Reshape(target_shape=(1, 1, self.configbuild["out_units"]), name='reshape_mapper'+self.suff.n)(x1)

        if self.up_type == "deconv":
            for i_layer in self.configbuild["BaseFilters"]:
                x1 = get_deconv(x1, i_layer, self.configbuild["kernel_size"], strides=(2, 2), padding="same", activation=self.configbuild["conv_act"], 
                        initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True, name_suffix=self.suff.n)
        
        elif self.up_type == "upsample":
            for i_layer in self.configbuild["BaseFilters"]:
                x1 = get_upsampling(x1, i_layer, self.configbuild["kernel_size"], strides=(1, 1), padding="same", up_size=(2, 2), 
                activation=self.configbuild["conv_act"], interpolation="nearest", initializer=self.initializer, add_noise=None, batch_norm=True, 
                                    name_suffix=self.suff.n)
                
        outputer = keras.layers.Conv2D(3, self.configbuild["kernel_size"], strides=(1, 1), padding="same", activation=self.configbuild["out_act"], use_bias=True,
                        name="BaseOutConv")(x1)
        assert tf.reduce_prod(x1.shape[1:-1]) == tf.reduce_prod(self.img_size), "mapping blocks do not correctly output shape, "\
        "must provide filters with length of {}".format(int(np.log2(self.img_size[0])))
        self.blocks.append(keras.Model(inputer, outputer, name="BaseGeneratorModel"))

    def _extend_model(self, filters=400, up=(1, 1), noise=False):
        '''extent one convolutional blocks with either deconvolution or upsampling'''
        if self.cur_img_size == self.targ_img_size:
            assert (up == (1, 1) and filters==3), "extent model cannot be called when image size is equal to target image size"

        if not self.initialized:
            raise ValueError("Generator model not initialized")
        inputer = keras.layers.Input(shape=self.blocks[-1].output_shape[1:], name="ExtendedInput"+self.suff.n)
        self.cur_img_size = (self.cur_img_size[0] * up[0], self.cur_img_size[0] * up[1])
        last_shape = self.blocks[-1].layers[-1].output_shape
        
        if self.up_type == "deconv":
            outputer = get_deconv(inputer, filters, self.configbuild["kernel_size"], strides=up, padding="same",
                activation=self.configbuild["conv_act"], initializer=self.initializer, add_noise=noise, max_norm=None, batch_norm=True, name_suffix=self.suff.n)
        
        elif self.up_type =="upsample":
            outputer = get_upsampling(inputer, filters, self.configbuild["kernel_size"], strides=(1, 1), padding="same", 
                up_size=up, activation=self.configbuild["conv_act"], interpolation="nearest", initializer=self.initializer, 
                add_noise=noise, batch_norm=True, name_suffix=self.suff.n)
            
        if up[0] > 1:
            outputer = keras.layers.Conv2D(3, (5, 5), strides=(1, 1), padding="same", activation=None, use_bias=True, name="Conv"+self.suff.n)(outputer)
            assert tf.reduce_prod(outputer.shape[1:-1]) == tf.reduce_prod(self.cur_img_size), "extended model do not correctly output shape"\
            "must provide filters with length of {}".format(int(np.log2(self.cur_img_size[0])))
        self.blocks.append(keras.Model(inputer, outputer))
        print("extended model from size of", last_shape, "to", outputer.shape)
    
    def _auto_extend(self):

        '''extend model automatically with arbitrary number of blocks provided in configbuild'''
        for ifilt in next(iter(self.configbuild["filters"]))[:-1]:
            self._extend_model(ifilt, up=(1, 1))
        self._extend_model(next(iter(self.configbuild["filters"]))[-1], up=(2, 2), noise=get_noise_out)

    def set_mapping_trainable(self, trainable=True, prefix='map_z'):
        for ly in self.get_flat_layers():
            if prefix in ly.name:
                ly.trainable = trainable
                print("set trainable_variables of layers {} to {}".format(ly.name, trainable))
    
    def set_joint_trainable(self, trainable=True, not_prefix='map_z'):
        for ly in self.get_flat_layers():
            if not_prefix not in ly.name and ly.trainable_variables != []:  
                ly.trainable = trainable
                print("set trainable_variables of layers {} to {}".format(ly.name, trainable))
                
    def forward_model(self, inputs, training=True, full_forw=True): ############ use get_model instead of calling function sequentially, this may result in async weights update
        full_forw = np.sum(self._n_call[:self._num_call_forw+1], dtype=np.int32) if not full_forw else None
        for bk in self.blocks[:]:
            inputs = bk(inputs)
        if self.apply_resize: # this however should be preprocess in datasets data handler
        #if (self.apply_resize and self.cur_img_size != self.targ_img_size):
            inputs = tf.image.resize(inputs, self.targ_img_size, method="nearest")
        return tf.cast(inputs, dtype=tf.float32)
    
    def get_model(self, get_with_functional=False, with_scope=False):
        assert not get_with_functional
        assert self.extendable, "This Generator is not extendable"
        U_layers = self.blocks if get_with_functional else self.get_flat_layers()
        inputer = keras.layers.Input(shape=(self.z_dim), name='BaseInput000')
        xi = U_layers[1](inputer)
        for J_layers in U_layers[2:]:
            xi = J_layers(xi)
        if with_scope: # this actually returns Model with scope since it is sync for all tf.Variable, Just in case 
            with self.strategy_scope.scope():
                return keras.Model(inputer, xi)
        else:
            return keras.Model(inputer, xi)
    
    def get_flat_layers(self):
        lys = []
        for msl in self.blocks:
            for ms in msl.layers: 
                lys.append(ms)
        return lys
    
    def print_config_layers(self):
        for i_layer in self.get_flat_layers():
            print("Name:{} - in_shape:{} - out_shape:{} - trainable:{} - scope{}".format(
            i_layer.name, i_layer.input_shape, i_layer.output_shape, i_layer.trainable, i_layer.name_scope()))
    
    def get_trainable(self, full_forw=False):
        # self.cur_trainable = self.get_model().trainable_variables
        self.cur_trainable = []
        if full_forw:
            for iu in self.get_flat_layers():
                for s in iu.trainable_variables: self.cur_trainable.append(s)
        else : 
            for iu in self.blocks[:np.sum(self._n_call[:self._num_call_forw+1], dtype=np.int32)]:
                for s in iu.trainable_variables:  self.cur_trainable.append(s)
        return self.cur_trainable

    def update_params(self, grads):
        assert self.optimizer is not None, "optimizer is not provided"
        if self.cur_trainable is None: # this must not run in the context of tf.function since it create new variables
            self.get_trainable()
        self.optimizer.apply_gradients(zip(grads, self.cur_trainable))

    def save_model(self, path):
        self.get_model().save(path)

class Discriminator_v1(BaseDiscriminator):
    
    def __init__(self, img_size=(32, 32), targ_img_size=(128, 128), seed=1010, strategy_scope=None, apply_resize=None, name="DISC02", configbuild={}):
        super(Discriminator_v1, self).__init__(img_size, targ_img_size , seed, strategy_scope, name, configbuild)
        self.cur_img_size = img_size
        self.suff = NameCaller()
        self.apply_resize = apply_resize

    def _initialize_base(self):

        cur_in = keras.layers.Input(shape=(*self.targ_img_size, 3))
        x1 = get_downsampling_conv(cur_in, self.configbuild["filters"][0][0], self.configbuild["kernel_size"], (1, 1), "same",  
                self.configbuild["conv_act"], None, self.initializer, add_noise=None, max_norm=None, batch_norm=True, name_suffix=self.suff.n)
        
        for drop in self.configbuild["filters"][1:-1]:
            
            x1 = get_downsampling_conv(x1, drop[0], self.configbuild["kernel_size"], (1, 1), "same",  
                    self.configbuild["conv_act"], drop[1], self.initializer, add_noise=None, max_norm=None, batch_norm=True, name_suffix=self.suff.n)
        
            cur_out = get_downsampling_conv(x1, 3, kernel_size=self.configbuild["kernel_size"], strides=(1, 1), padding="same", 
                        activation='relu', pooling=drop[1], initializer=self.initializer, add_noise=None, max_norm=None, batch_norm=True, name_suffix=self.suff.n)
           
        x1 = keras.layers.Flatten(name="flatten_d_conv")(x1)
        for ui in self.configbuild["units_dense"]:
            x1 = keras.layers.Dense(ui, activation=self.configbuild["dense_act"], use_bias=True, kernel_initializer=self.initializer, name="Dense"+self.suff.n)(x1)
        cur_out = keras.layers.Dense(1, activation=self.configbuild["out_act"], name="OutDense"+self.suff.n)(x1)
        self.model = keras.Model(cur_in, cur_out, name="DiscModel_"+self.name)
        
    def forward_model(self, inputs, training=True):
        if (self.cur_img_size != self.targ_img_size and self.apply_resize): # fixed model, upsample first
            shapex = inputs.shape[1:-1][0]
            multiplier = int(self.targ_img_size[0] / shapex)
            if multiplier > 1:
                inputs = keras.layers.UpSampling2D((multiplier, multiplier), interpolation="nearest")(inputs)  
        return self.model(inputs, training=training)
    
    def get_trainable(self):
        return self.model.trainable_variables

    def update_params(self, grads):
        assert self.optimizer is not None, "optimizer is not provided"
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def save_model(self, path):
        self.model.save(path)

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