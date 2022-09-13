import tensorflow as tf
import numpy as np
from keras.layers import Layer, Input, Dense, Conv2D, UpSampling2D, Reshape, LeakyReLU, Activation, Flatten, MaxPooling2D
from keras.activations import relu, selu
from keras.initializers import GlorotNormal, RandomNormal
from tensorflow_addons.layers import InstanceNormalization

from utils import NameCaller, function_call

class AdaIN(Layer):

    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.w_channels = input_shapes[1][-1]
        self.x_channels = input_shapes[0][-1]
        self.dense_1 = Dense(self.x_channels)
        self.dense_2 = Dense(self.x_channels)

    def call(self, inputs):
        _x0, _w0 = inputs
        ys = tf.reshape(self.dense_1(_w0), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(_w0), (-1, 1, 1, self.x_channels))
        return ys * _x0 + yb

class MinSTD(Layer):

    def __init__(self, group_size=4, **kwargs):
        super(MinSTD, self).__init__(**kwargs)
        self.group_size = group_size

    def call(self, inputs):
        with tf.name_scope('MinibatchStddev'):
            group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])    
            s = inputs.shape                                             
            y = tf.cast(tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3]]), dtype=tf.float32)                         
            y -= tf.reduce_mean(y, axis=0, keepdims=True)           
            y = tf.sqrt(tf.reduce_mean(tf.square(y), axis=0) + 1e-8)              
            y = tf.cast(tf.reduce_mean(y, axis=[1,2,3], keepdims=True), inputs.dtype)                                 
            y = tf.tile(y, [group_size, 1, s[2], s[3]])      

        return tf.concat([inputs, y], axis=1)                        

class Noise(Layer):
    
    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(
            shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel"
        )

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output

class PixNorm(Layer):

    def __init__(self, **kwargs):
        super(PixNorm, self).__init__(**kwargs)

    def call(self, inputs):
        with tf.name_scope('PixelNorm'):
            return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + 1e-8)

class Interpolate(Layer):

    def __init__(self, **kwargs):
        super(Interpolate, self).__init__(**kwargs)

    def call(self, inputs):
        x, y, alpha = inputs
        return x + alpha * (y - x)
    
def get_noise(distribution, n):
    with tf.name_scope("sampled_noise"):
        return distribution.sample(n)

class Generator:
    '''Generator class
    Inputs :
        BuildConfig : BuildConfig object
        save_path : path to save the model
        strategy : tf distributed strategy to be used
    '''
    def __init__(self, BuildConfig, save_path="", strategy=None):
        self.BuildConfig = BuildConfig
        self._strategy = strategy
        self._init_res_log2 = int(np.log2(self.BuildConfig.img_size[0]))
        self._initializer = None
        self._cur_img_size = 2 ** self._init_res_log2 # this will keep update when model is extended
        self._out_map_shape = self.BuildConfig.G_dense_units[-1]
        self._layer_n = NameCaller()
        self._default_graph = None
        self._is_initialized = False
        self._initialized_model = None
        self.save_path = ""
        self._model = 45455

    def initialize_base(self):
        if self._strategy is None:
            self._initialize_base()
        else : 
            with self._strategy.scope():
                self._initialize_base()
                print("Initialize Generator with strategy")
                
    def _initialize_base(self):
        assert not self._is_initialized, "Generator is already initialized"
        n_inputs = [] # list noise input
        rgb_outs = []
        if self._initializer is None:
            self._initializer = self.BuildConfig.initializer if self.BuildConfig.initializer is not None else GlorotNormal(self.BuildConfig.seed)
        if (self.BuildConfig.G_conv_act == "LeakyReLU" and self.BuildConfig.G_dense_act == "LeakyReLU"): 
            conv_act = LeakyReLU
            dense_act = LeakyReLU
        else:
            conv_act = Activation(self.BuildConfig.G_conv_act)
            dense_act = Activation(self.BuildConfig.G_dense_act)

        Ft = self.BuildConfig.G_filters # dict map log2(res) to filter
        res = 2 ** self._init_res_log2 # initial resolution
        with tf.name_scope("LatentMapping"):
            map_input = Input(shape=self.BuildConfig.latent_dim, dtype=tf.float32, name='map_input'+self._layer_n.n)
            x_ = map_input
            for unit in self.BuildConfig.G_dense_units:
                x_ = Dense(unit, kernel_initializer=self._initializer, name='dense_map'+self._layer_n.n)(x_)
                x_ = dense_act(name="activation"+self._layer_n.n)(x_)
                x_ = PixNorm(name="pixelnorm"+self._layer_n.n)(x_)
            map_out = Reshape((1, 1, int(np.prod(x_.shape[1:]))), name='reshape_map'+self._layer_n.n)(x_)
        
        with tf.name_scope("ConstConv"): # Base Conv 4x4
            const_input = Input(shape=(res, res, Ft[res]), name='const_input'+self._layer_n.n)
            noise_input = Input(shape=(res, res, 1), dtype=tf.float32, name='noise_input'+self._layer_n.n) # from convolution
            n_inputs.append(noise_input)
            for _ in range(2):
                x_ = Conv2D(Ft[res], self.BuildConfig.G_kernel_size, (1, 1), padding="same", kernel_initializer=self._initializer, name='conv_up'+self._layer_n.n)(const_input)
                x_ = Noise(trainable=True, name="Noise"+self._layer_n.n)([x_, noise_input])
                x_ = conv_act(name="activation"+self._layer_n.n)(x_)
                x_ = AdaIN()([x_, map_out]) # shape ([None, res, res, 3], [None, map_output])
                
        def extend_rgb(x_k):
            shape_rgb = str(x_k.shape[1])
            with tf.name_scope("RGBTrans{}x{}".format(shape_rgb, shape_rgb)):
                x_k = Conv2D(3, kernel_size=self.BuildConfig.G_kernel_size, padding="same", kernel_initializer=self._initializer, name='rgb'+self._layer_n.n)(x_k)
            return x_k

        def extend_model(inputs, filter, res):
            # 0 : conv, 1 : noise, 2 : mapping
            with tf.name_scope("UpConv{}x{}".format(res, res)):
                x_ = UpSampling2D(size=(2, 2), interpolation='nearest', name="upsampling"+self._layer_n.n)(inputs[0])
                for _ in range(2):
                    x_ = Conv2D(filter, self.BuildConfig.G_kernel_size, strides=(1, 1), padding="same", kernel_initializer=self._initializer, name=''+self._layer_n.n)(x_)
                    x_ = Noise(trainable=True, name="Noise"+self._layer_n.n)([x_, inputs[1]]) # noise input shape must be updated
                    x_ = conv_act(name="activation"+self._layer_n.n)(x_)
                    x_ = AdaIN()([x_, inputs[2]])
                    x_ = InstanceNormalization(axis=-1, center=True, scale=True, name='InstanceNorm'+self._layer_n.n)(x_)
            return x_

        x_rgb_0 = extend_rgb(x_)
        rgb_outs.append(x_rgb_0)
        x_rgb_0 = UpSampling2D((2, 2), name="up_rgb"+self._layer_n.n)(x_rgb_0)

        for i_ in range(int(np.log2(self.BuildConfig.targ_img_size[0])) - self._init_res_log2):
            if i_ != 0:
                x_rgb_0 = UpSampling2D((2, 2), name="up_rgb"+self._layer_n.n)(x_rgb_0)
            res = 2 * res
            noise_input = Input(shape=(res, res, 1), dtype=tf.float32, name='noise_input'+self._layer_n.n) # from convolution
            n_inputs.append(noise_input)
            x_ = extend_model([x_, noise_input, map_out], self.BuildConfig.G_filters[3], res)
            x_rgb_1 = extend_rgb(x_)
            
            with tf.name_scope("lerp"):
                x_rgb_0 = Interpolate(name="interpolate"+self._layer_n.n)([x_rgb_0, x_rgb_1, self.BuildConfig.lerp])
                rgb_outs.append(x_rgb_0)

        self._is_initialized = True
        self._initialized_model = (n_inputs, map_input, const_input, rgb_outs)

    @property
    def stages(self):
        return int(np.log2(self.BuildConfig.targ_img_size[0])) - self._init_res_log2 + 1
    
    def _initialize_as_graph(self):
        '''Get graph, should not be called directly'''
        assert not self._is_initialized, "Generator is already initialized"
        self._default_graph = tf.Graph()
        with self._default_graph.as_default(): 
            self._initialize_base()

    def _get_easy_model(self, targ_res=None):
        '''get model this provided targ_res which can be access via Generator._model'''
        assert self._is_initialized, "Generator is not initialized"
        if targ_res is not None : assert type(targ_res) == int, 'targ_res must be int'
        if targ_res == None: targ_res = self._cur_img_size
        idx_layers = int(np.log2(targ_res)) - self._init_res_log2 + 1
        with tf.name_scope("SimpleGenerator"):
            n_inputs, map_input, const_input, rgb_outs = self._initialized_model
            self._model = tf.keras.Model([*n_inputs[:idx_layers], map_input, const_input], rgb_outs[:idx_layers])

    def forward_model(self, inputs, training=True, targ_res=None):
        '''Make inference to Generator
        inputs (list) : input to the model
        training (bool) : training mode
        targ_res (int) : target resolution
        '''
        if targ_res is None : targ_res = self._cur_img_size
        if targ_res != self._model.input_shape[-1][1] or self._model is None:
            self._get_easy_model(targ_res=targ_res)
        return self._model(inputs, training=training)
    
    def get_snap_rgb(self, inputs, targ_res=None):
        '''Get snapshot of rgb image
        inputs (list) : input to the model
        targ_res (int) : target resolution
        '''
        if targ_res is None : targ_res = self._cur_img_size
        return self.forward_model(inputs, False, targ_res=targ_res)[-1]
        
    def save_model(self, targ_res=None):
        '''Save_model'''
        if targ_res is None : targ_res = self._cur_img_size
        self._get_easy_model(targ_res=targ_res)
        _save_path = self.save_path + "{}x{}".format(targ_res, targ_res)
        self._model.save(_save_path) 
        print("Saved Generator of {}x{} to {}".format(targ_res, targ_res, _save_path))

    def load_partial(self, path_to_model):
        '''Load partial model which will be used in transfer learning for current model. 
        The loaded model can be smaller than current model which will only load the layers that are in common.'''
        assert self._is_initialized, "Generator is not initialized"
        _loaded_model = tf.keras.models.load_model(path_to_model)
        _out_shape = _loaded_model.output_shape[-1][1]
        _len_var = len(_loaded_model.trainable_variables)
        self._get_easy_model(_out_shape)
        self._model.trainable_variables[:_len_var] = _loaded_model.trainable_variables
        print("Loaded Generator of {}x{} from {}".format(_out_shape, _out_shape, path_to_model))

    def get_trainable(self, targ_res=None):
        '''Get trainable variables'''
        self._get_easy_model(targ_res=targ_res)
        return self._model.trainable_variables

    @property
    def input_shape(self):
        assert self._model is not None, "Generator is not initialized"
        return [i[1:] for i in self._model.input_shape]
    
class Discriminator:
    '''Discriminator class
    Inputs : 
        BuildConfig : BuildConfig class
        exhuastive : either 'True', 'False' or 'full'. Note that user must provided as False.
        save_path : path to save model
        strategy : tf distributed strategy to be used
    '''
    def __init__(self, BuildConfig, save_path="", strategy=None):

        self.BuildConfig = BuildConfig
        self._strategy = strategy
        self._init_res_log2 = int(np.log2(self.BuildConfig.img_size[0]))
        self._initializer = None
        self._cur_img_size = 2 ** self._init_res_log2 # this will keep update when model is extended
        self._layer_n = NameCaller()
        self._default_graph = None
        self._is_initialized = False
        self._initialized_model = None
        self.save_path = ""
        self.exhaustive = False
        self._model = None
        
    def initialize_base(self):
        if self._strategy is None:
            self._initialize_base()
        else : 
            with self._strategy.scope():
                self._initialize_base()
                print("Initialize Discriminator with strategy")
                
    def _initialize_base(self):
        assert not self._is_initialized, "Discriminator is already initialized"
        if self._initializer is None:
            self._initializer = self.BuildConfig.initializer if self.BuildConfig.initializer is not None else GlorotNormal(self.BuildConfig.seed)
        if (self.BuildConfig.D_conv_act == "LeakyReLU" and self.BuildConfig.D_dense_act == "LeakyReLU"): 
            conv_act = LeakyReLU
            dense_act = LeakyReLU
        else:
            conv_act = Activation(self.BuildConfig.D_conv_act)
            dense_act = Activation(self.BuildConfig.D_dense_act)
        if not self.exhaustive:

            Ft = self.BuildConfig.D_filters
            rgb_ins, blocks = [], []
            def li(_r): return int(np.log2(_r))
        
            def _f_rgb(res_, filters):
                ins = Input((res_, res_, 3), name="rgb_in"+self._layer_n.n)
                outs = Conv2D(filters, self.BuildConfig.D_kernel_size, padding="same", kernel_initializer=self._initializer, 
                                            name="rgb_conv{}x{}".format(res_, res_)+self._layer_n.n)(ins)
                outs = conv_act(name="act{}x{}".format(res, res)+self._layer_n.n)(outs)
                return tf.keras.Model(ins, outs)
            
            def _f_base(filters0, res_):
                ins = Input((res_, res_, filters0), name="conv_in"+self._layer_n.n)
                x_k = MinSTD(group_size=4)(ins)
                x_k = Conv2D(filters0, self.BuildConfig.D_kernel_size, padding="same", kernel_initializer=self._initializer, 
                                            name="conv{}x{}".format(res_, res_)+self._layer_n.n)(x_k)
                x_k = conv_act(name="act{}x{}".format(res, res)+self._layer_n.n)(x_k)
                
                with tf.name_scope("dense_blocks"):
                    x_k = Flatten(name="flatten"+self._layer_n.n)(x_k)
                    for unit in self.BuildConfig.D_dense_units:
                        x_k = Dense(unit, kernel_initializer=self._initializer, name="dense"+self._layer_n.n)(x_k)
                        x_k = dense_act(name="act"+self._layer_n.n)(x_k)
                    x_k = Dense(1, kernel_initializer=self._initializer, name="dense"+self._layer_n.n)(x_k)
                return tf.keras.Model(ins, x_k)
            
            def _f_conv_bloc(res_, filters0, filters1, minstd=False):
                ins = Input((res_, res_, filters0), name="conv_in"+self._layer_n.n)
                if minstd: x_k = MinSTD(group_size=4)(ins)
                else:
                     x_k = ins
                for filt in [filters0, filters1]:
                    x_k = Conv2D(filt, self.BuildConfig.D_kernel_size, padding="same", kernel_initializer=self._initializer, 
                                                name="conv{}x{}".format(res_, res_)+self._layer_n.n)(x_k)
                    x_k = conv_act(name="act{}x{}".format(res, res)+self._layer_n.n)(x_k)
                x_k = MaxPooling2D((2, 2), name="pool{}x{}".format(res_, res_)+self._layer_n.n)(x_k)
                return tf.keras.Model(ins, x_k)
            
            with tf.name_scope("DiscrimConv"):
                for r_ in range(self._init_res_log2, li(self.BuildConfig.targ_img_size[0]) + 1):
                    res = 2 ** r_
                    with tf.name_scope("conv_block{}x{}".format(res, res)):
                        rgb_ins.append(_f_rgb(res_=res, filters=Ft[li(res)]))
                        if r_ == self._init_res_log2:
                            blocks.append(_f_base(Ft[li(res)], res))
                        else:
                            blocks.append(_f_conv_bloc(res, Ft[li(res)], Ft[li(res) - 1]))                
        else:
            raise NotImplementedError
        self._is_initialized = True
        self._initialized_model = (rgb_ins, blocks)
    
    def _initialize_as_graph(self):
        '''Get graph, should not be called directly'''
        assert not self._is_initialized, "Discriminator is already initialized"
        self._default_graph = tf.Graph()
        with self._default_graph.as_default():
            _ = self._initialize_base()
            
    def _get_easy_model(self, targ_res=None):
        '''get model this provided targ_res which can be access via Generator._model'''
        assert self._is_initialized, "Discriminator is not initialized"
        if targ_res == None: targ_res = self._cur_img_size
        _res = 2 ** targ_res
        _idx = int(np.log2(targ_res)) - self._init_res_log2
        _alp = Input(shape=(1), name="alpha_lerp"+self._layer_n.n)
        _img_in = Input(shape=(targ_res, targ_res, 3), name='img_in'+self._layer_n.n)
        x_rgb_0 = function_call(self._initialized_model[0][_idx], _img_in)
        x_rgb_0 = function_call(self._initialized_model[1][_idx], x_rgb_0)
        
        if _idx > 0:
            _idx -= 1
            x_rgb_1 = MaxPooling2D((2, 2), name="pool{}x{}".format(targ_res, targ_res)+self._layer_n.n)(_img_in)
            x_rgb_1 = function_call(self._initialized_model[0][_idx], x_rgb_1)

            with tf.name_scope("lerp"):
                x_rgb_0 = Interpolate(name="interpolate"+self._layer_n.n)([x_rgb_0, x_rgb_1, self.BuildConfig.lerp])
            for i in range(_idx, -1, -1):
                x_rgb_0 = function_call(self._initialized_model[1][i], x_rgb_0)
        with tf.name_scope("SimpleDiscriminator"):
            self._model = tf.keras.Model([_img_in, _alp], x_rgb_0)

    def forward_model(self, input, training=True, targ_res=None):
        '''Make inference to Discriminator
            inputs (list) : input to the model
            training (bool) : training mode
            targ_res (int) : target resolution
        '''
        if targ_res is None : targ_res = self._cur_img_size
        if targ_res != self._model.input_shape[0][1] or self._model is None:
            self._get_easy_model(targ_res)
        return self._model(input, training=training)

    def save_model(self, targ_res=None):
        '''Save_model'''
        if targ_res is None : targ_res = self._cur_img_size
        self._get_easy_model(targ_res=targ_res)
        _save_path = self.save_path + "{}x{}".format(targ_res, targ_res)
        self._model.save(_save_path)
        print("Saved Discriminator of {}x{} to {}".format(targ_res, targ_res, _save_path))

    def load_partial(self, path_to_model):
        '''Load partial model which will be used in transfer learning for current model. 
        The loaded model can be smaller than current model which will only load the layers that are in common.'''
        assert self._is_initialized, "Discriminator is not initialized"
        _loaded_model = tf.keras.models.load_model(path_to_model)
        _out_shape = _loaded_model.output_shape[-1][1]
        _len_var = len(_loaded_model.trainable_variables)
        self._get_easy_model(_out_shape)
        self._model.trainable_variables[:_len_var] = _loaded_model.trainable_variables
        print("Loaded Discriminator of {}x{} from {}".format(_out_shape, _out_shape, path_to_model))

    def get_trainable(self, targ_res=None):
        '''Get trainable variables'''
        self._get_easy_model(targ_res=targ_res)
        return self._model.trainable_variables
    
    @property
    def input_shape(self):
        assert self._model is not None, "Discriminator is not initialized"
        return [i[1:] for i in self._model.input_shape]